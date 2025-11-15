import os
import csv
import argparse
import psycopg2
from psycopg2.extras import execute_values


def get_conn():
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ["POSTGRES_DB"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return psycopg2.connect(dbname=db, user=user, password=password, host=host, port=port)


def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == '' or s.upper() == 'NULL':
        return None
    s = s.replace('%', '')
    s = s.replace(',', '.')
    try:
        return float(s)
    except Exception:
        return None


def safe_int_in_range(value, min_v=0, max_v=10):
    if value is None or value == '':
        return None
    s = str(value).strip()
    if s == '':
        return None
    try:
        iv = int(float(s)) if ('.' in s or ',' in s) else int(s)
    except Exception:
        return None
    if iv < min_v or iv > max_v:
        return None
    return iv


def import_upcoming(csv_path, gametype, year, week, replace=False):
    # Read rows in given order (1..N as matchnummer)
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.get_dialect('excel')
        reader = csv.DictReader(f, dialect=dialect)
        for r in reader:
            rows.append(r)

    datatype = 'deeppick'

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Check if an identical upcoming round already exists
            cur.execute(
                """
                SELECT omgang_id FROM omgang
                WHERE year = %s AND week = %s AND datatype = %s AND gametype = %s
                """,
                (year, week, datatype, gametype),
            )
            existing = cur.fetchone()
            if existing:
                omgang_id = existing[0]
                print(f"Omgång finns redan (omgang_id={omgang_id}). {'Ersätter matcher' if replace else 'Hoppar insert av matcher'}.")
            else:
                cur.execute(
                    """
                    INSERT INTO omgang (year, week, datatype, svspelinfo_id, gametype)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING omgang_id
                    """,
                    (year, week, datatype, None, gametype),
                )
                omgang_id = cur.fetchone()[0]
                print(f"Skapade ny omgång (omgang_id={omgang_id}) för {gametype} {year}-v{week}")

            if replace and existing:
                cur.execute("DELETE FROM omgangsmatch WHERE omgang_id = %s", (omgang_id,))

            match_values = []
            oddset_sum_max = 0.0
            oddset_sum_min = 0.0
            svfolk_sum_max = 0.0
            svfolk_sum_min = 0.0

            def derive_people_odds(p1, px, p2):
                vals = [v for v in (p1, px, p2) if v is not None and v > 0]
                if len(vals) == 0:
                    return None, None, None
                # Normalisera till sannolikheter som summerar till 1.0
                total = sum(vals)
                # Om total råkar vara 0 (alla nollor), returnera None
                if total <= 0:
                    return None, None, None
                # Beräkna pseudo-odds ~ 1 / p
                def inv_or_none(v):
                    return (1.0 / (v / total)) if (v is not None and v > 0) else None
                o1, ox, o2 = inv_or_none(p1), inv_or_none(px), inv_or_none(p2)
                # Avrunda till två decimaler för läsbarhet
                def r2(x):
                    return round(x, 2) if x is not None else None
                return r2(o1), r2(ox), r2(o2)

            for idx, r in enumerate(rows, start=1):
                hemmalag = r.get('home_team') or r.get('hemmalag')
                bortalag = r.get('away_team') or r.get('bortalag')

                oddset1 = to_float(r.get('oddset1'))
                oddsetx = to_float(r.get('oddsetx'))
                oddset2 = to_float(r.get('oddset2'))

                oddset_procent1 = to_float(r.get('oddset_procent1'))
                oddset_procentx = to_float(r.get('oddset_procentx'))
                oddset_procent2 = to_float(r.get('oddset_procent2'))

                svenska_folket_procent1 = to_float(r.get('svenska_folket1'))
                svenska_folket_procentx = to_float(r.get('svenska_folketx'))
                svenska_folket_procent2 = to_float(r.get('svenska_folket2'))

                # Härled people-odds från procent (monotont: högre % -> lägre odds)
                svenska_folket_odds1, svenska_folket_oddsx, svenska_folket_odds2 = derive_people_odds(
                    svenska_folket_procent1, svenska_folket_procentx, svenska_folket_procent2
                )

                oddset_rank = int(to_float(r.get('oddset_rank'))) if r.get('oddset_rank') not in (None, '') else None
                people_rank = int(to_float(r.get('people_rank'))) if r.get('people_rank') not in (None, '') else None

                # Tio Tidningar: primary keys without underscore, fallback to old underscore names
                tio_tidningar1 = safe_int_in_range(r.get('tiotidningar1') or r.get('tiotidningar_1'))
                tio_tidningarx = safe_int_in_range(r.get('tiotidningarx') or r.get('tiotidningar_x'))
                tio_tidningar2 = safe_int_in_range(r.get('tiotidningar2') or r.get('tiotidningar_2'))

                sum_tio = sum([v for v in (tio_tidningar1, tio_tidningarx, tio_tidningar2) if v is not None])
                tio_tidningar_present = True if sum_tio > 0 else False
                if not tio_tidningar_present:
                    tio_tidningar1 = None
                    tio_tidningarx = None
                    tio_tidningar2 = None

                o_vals = [v for v in (oddset_procent1, oddset_procentx, oddset_procent2) if v is not None]
                if o_vals:
                    oddset_sum_max += max(o_vals)
                    oddset_sum_min += min(o_vals)
                s_vals = [v for v in (svenska_folket_procent1, svenska_folket_procentx, svenska_folket_procent2) if v is not None]
                if s_vals:
                    svfolk_sum_max += max(s_vals)
                    svfolk_sum_min += min(s_vals)

                match_values.append((
                    omgang_id,
                    idx,
                    hemmalag,
                    bortalag,
                    oddset_rank,
                    people_rank,
                    oddset1,
                    oddsetx,
                    oddset2,
                    oddset_procent1,
                    oddset_procentx,
                    oddset_procent2,
                    svenska_folket_odds1,
                    svenska_folket_oddsx,
                    svenska_folket_odds2,
                    svenska_folket_procent1,
                    svenska_folket_procentx,
                    svenska_folket_procent2,
                    tio_tidningar1,
                    tio_tidningarx,
                    tio_tidningar2,
                    tio_tidningar_present,
                ))

            if match_values:
                insert_sql = """
                INSERT INTO omgangsmatch (
                    omgang_id, matchnummer, hemmalag, bortalag, oddset_rank, people_rank,
                    oddset1, oddsetx, oddset2,
                    oddset_procent1, oddset_procentx, oddset_procent2,
                    svenska_folket_odds1, svenska_folket_oddsx, svenska_folket_odds2,
                    svenska_folket_procent1, svenska_folket_procentx, svenska_folket_procent2,
                    tio_tidningar1, tio_tidningarx, tio_tidningar2,
                    tio_tidningar_present
                ) VALUES %s
                ON CONFLICT (omgang_id, matchnummer) DO NOTHING
                """
                execute_values(cur, insert_sql, match_values)

            # Spara radsummor som heltal (inga decimaler i omgang‑tabellen)
            o_max_i = int(round(oddset_sum_max)) if oddset_sum_max is not None else None
            o_min_i = int(round(oddset_sum_min)) if oddset_sum_min is not None else None
            s_max_i = int(round(svfolk_sum_max)) if svfolk_sum_max is not None else None
            s_min_i = int(round(svfolk_sum_min)) if svfolk_sum_min is not None else None
            cur.execute(
                """
                UPDATE omgang
                SET oddset_radsumma_max = %s,
                    oddset_radsumma_min = %s,
                    svenska_folket_radsumma_max = %s,
                    svenska_folket_radsumma_min = %s
                WHERE omgang_id = %s
                """,
                (o_max_i, o_min_i, s_max_i, s_min_i, omgang_id),
            )

            conn.commit()
            print(f"Importerade {len(match_values)} matcher till omgång {omgang_id} ({gametype} {year}-v{week}).")


def main():
    parser = argparse.ArgumentParser(description='Importera kommande omgång från CSV (deeppick-format)')
    parser.add_argument('csvfile', help='Sökväg till CSV-filen för omgången (t.ex. svenskaspel/Stryk-25-10-1-*.csv)')
    parser.add_argument('--gametype', required=True, help='Speltyp, t.ex. Stryktipset/Europatipset')
    parser.add_argument('--year', type=int, required=True, help='År för omgången (YYYY)')
    parser.add_argument('--week', type=int, required=True, help='Vecka för omgången (WW)')
    parser.add_argument('--replace', action='store_true', help='Ersätt befintliga matchrader (om omgången redan finns)')
    args = parser.parse_args()

    import_upcoming(args.csvfile, args.gametype, args.year, args.week, replace=args.replace)


if __name__ == '__main__':
    main()
