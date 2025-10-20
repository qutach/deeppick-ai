import os
import csv
import argparse
from collections import defaultdict
import psycopg2
from psycopg2.extras import execute_values


def get_conn():
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ["POSTGRES_DB"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return psycopg2.connect(dbname=db, user=user, password=password, host=host, port=port)


def parse_omg_to_year_week(omg_value):
    # Expecting format like 202541 -> year=2025 week=41
    s = str(omg_value)
    if len(s) >= 6:
        year = int(s[:4])
        week = int(s[4:])
        return year, week
    # fallback: try to split by - or /
    parts = s.split("-")
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return None, None


def ensure_kombination(cur, rad):
    cur.execute("SELECT kombinations_id FROM kombinationer WHERE rad = %s", (rad,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO kombinationer (rad) VALUES (%s) RETURNING kombinations_id", (rad,))
    return cur.fetchone()[0]


def import_file(csv_path, max_omgangar=None, replace=False):
    # Read CSV and group rows by (produktnamn, omg, svspelinfo_id)
    groups = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8') as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.get_dialect('excel')
        reader = csv.DictReader(f, dialect=dialect)

        def get_field(d, *names):
            for n in names:
                if n in d and d[n] not in (None, ''):
                    return d[n]
            return None

        for r in reader:
            key = (
                get_field(r, 'produktnamn', 'produkt_namn', 'produkt'),
                get_field(r, 'omg', 'omgang', 'om'),
                get_field(r, 'svspelinfo_id'),
            )
            groups[key].append(r)

    processed = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for (produkt, omg, svspelinfo_id_key), rows in groups.items():
                if max_omgangar is not None and processed >= max_omgangar:
                    break

                # Determine year/week from 'omg' column
                year, week = parse_omg_to_year_week(omg)
                if year is None or week is None:
                    print(f"Hoppar omgång pga okänt oms-format: produkt={produkt} omg={omg} (rådata första rad: {rows[0]})")
                    processed += 1
                    continue
                # For this import source, datatype is static
                datatype = 'tipsextra'
                # Use produkt as gametype (produkt innehåller t.ex. 'Europatipset' eller 'Stryktipset')
                gametype = (produkt or '').strip()
                # Use svspelinfo_id from grouping key (fallback to first row)
                svspelinfo_id_raw = svspelinfo_id_key if svspelinfo_id_key is not None else rows[0].get('svspelinfo_id')
                svspelinfo_id = int(svspelinfo_id_raw) if (svspelinfo_id_raw is not None and str(svspelinfo_id_raw).strip() != '') else None

                # Check if an identical omgang already exists: match on year, week, datatype, svspelinfo_id
                cur.execute(
                    "SELECT omgang_id FROM omgang WHERE year = %s AND week = %s AND datatype = %s AND svspelinfo_id = %s",
                    (year, week, datatype, svspelinfo_id),
                )
                existing = cur.fetchone()
                if existing:
                    print(f"Omgång redan finns (omgang_id={existing[0]}). Hoppar över: {produkt} {omg}")
                    processed += 1
                    continue

                # Insert into omgang
                cur.execute(
                    "INSERT INTO omgang (year, week, datatype, svspelinfo_id, gametype) VALUES (%s, %s, %s, %s, %s) RETURNING omgang_id",
                    (year, week, datatype, svspelinfo_id, gametype),
                )
                omgang_id = cur.fetchone()[0]
                print(f"Skapar omgang {omgang_id} för {produkt} {omg} (year={year}, week={week})")

                # Prepare omgangsmatch inserts
                match_values = []
                # Summeringar (max/min per match) för Oddset och Svenska Folket
                oddset_sum_max = 0.0
                oddset_sum_min = 0.0
                svfolk_sum_max = 0.0
                svfolk_sum_min = 0.0
                for r in rows:
                    # convert types, handle empty strings
                    matchnummer = int(r.get('matchnummer') or 0)
                    hemmalag = r.get('hemmalag')
                    bortalag = r.get('bortalag')
                    oddset_rank = int(r.get('oddset_rank')) if r.get('oddset_rank') else None
                    people_rank = int(r.get('people_rank')) if r.get('people_rank') else None
                    def to_float(x):
                        # Accept empty/NULL, strip percent sign, and handle both dot and comma decimal separators
                        if x is None:
                            return None
                        s = str(x).strip()
                        if s == '' or s.upper() == 'NULL':
                            return None
                        # remove percent sign if present (we don't want '%' in stored value)
                        s = s.replace('%', '')
                        # normalize comma as decimal separator
                        s = s.replace(',', '.')
                        try:
                            return float(s)
                        except Exception:
                            print(f"Varning: kunde inte konvertera värdet till float: '{x}'")
                            return None

                    oddset1 = to_float(r.get('oddset1'))
                    oddsetx = to_float(r.get('oddsetx'))
                    oddset2 = to_float(r.get('oddset2'))
                    oddset_procent1 = to_float(r.get('oddset_procent1'))
                    oddset_procentx = to_float(r.get('oddset_procentx'))
                    oddset_procent2 = to_float(r.get('oddset_procent2'))
                    svenska_folket_odds1 = to_float(r.get('svenska_folket_odds1'))
                    svenska_folket_oddsx = to_float(r.get('svenska_folket_oddsx'))
                    svenska_folket_odds2 = to_float(r.get('svenska_folket_odds2'))
                    # Prefer the 'procent' columns; if they're missing or empty, leave as NULL
                    svenska_folket_procent1 = to_float(get_field(r, 'svenska_folket1'))
                    svenska_folket_procentx = to_float(get_field(r, 'svenska_folketx'))
                    svenska_folket_procent2 = to_float(get_field(r, 'svenska_folket2'))

                    # Uppdatera radsummor (max/min per match)
                    o_vals = [v for v in (oddset_procent1, oddset_procentx, oddset_procent2) if v is not None]
                    if o_vals:
                        oddset_sum_max += max(o_vals)
                        oddset_sum_min += min(o_vals)
                    s_vals = [v for v in (svenska_folket_procent1, svenska_folket_procentx, svenska_folket_procent2) if v is not None]
                    if s_vals:
                        svfolk_sum_max += max(s_vals)
                        svfolk_sum_min += min(s_vals)
                    def safe_int_in_range(value, min_v=0, max_v=10):
                        if value is None or value == '':
                            return None
                        s = str(value).strip()
                        # Accept empty after strip
                        if s == '':
                            return None
                        try:
                            iv = int(float(s)) if ('.' in s or ',' in s) else int(s)
                        except Exception:
                            print(f"Varning: ogiltigt tio_tidningar-värde '{value}' — sätter NULL")
                            return None
                        if iv < min_v or iv > max_v:
                            print(f"Varning: tio_tidningar-värde {iv} utanför [{min_v},{max_v}] — sätter NULL")
                            return None
                        return iv

                    tio_tidningar1 = safe_int_in_range(r.get('tio_tidningar1'))
                    tio_tidningarx = safe_int_in_range(r.get('tio_tidningarx'))
                    tio_tidningar2 = safe_int_in_range(r.get('tio_tidningar2'))

                    # Compute presence flag: True if any of the three is non-null and sum>0
                    sum_tio = sum([v for v in (tio_tidningar1, tio_tidningarx, tio_tidningar2) if v is not None])
                    tio_tidningar_present = True if sum_tio > 0 else False
                    # If absent (sum == 0) treat the individual fields as NULL
                    if not tio_tidningar_present:
                        tio_tidningar1 = None
                        tio_tidningarx = None
                        tio_tidningar2 = None

                    match_values.append((
                        omgang_id,
                        matchnummer,
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

                # Optionally remove existing rows for this omgang (if replace=True)
                if replace:
                    cur.execute("DELETE FROM omgangsmatch WHERE omgang_id = %s", (omgang_id,))

                # Bulk insert into omgangsmatch (ignore conflicts on (omgang_id, matchnummer))
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

                # Uppdatera radsummor på omgang
                cur.execute(
                    """
                    UPDATE omgang
                    SET oddset_radsumma_max = %s,
                        oddset_radsumma_min = %s,
                        svenska_folket_radsumma_max = %s,
                        svenska_folket_radsumma_min = %s
                    WHERE omgang_id = %s
                    """,
                    (oddset_sum_max, oddset_sum_min, svfolk_sum_max, svfolk_sum_min, omgang_id),
                )

                # Build combination string ordered by matchnummer
                rows_sorted = sorted(rows, key=lambda rr: int(rr.get('matchnummer') or 0))
                outcome = ''.join([ (rr.get('utfall') or '').strip()[:1].upper() for rr in rows_sorted ])
                if len(outcome) == 0:
                    print(f"Varning: kunde inte bygga 'correct' för omgang {omg} (produkt {produkt})")
                else:
                    # Ensure combination exists and update omgang.correct
                    komb_id = ensure_kombination(cur, outcome)
                    cur.execute("UPDATE omgang SET correct = %s WHERE omgang_id = %s", (komb_id, omgang_id))
                    print(f"Uppdaterade omgang.correct -> kombinations_id={komb_id} (rad={outcome})")

                conn.commit()
                processed += 1

    print(f"Importerade {processed} omgång(ar).")


def main():
    parser = argparse.ArgumentParser(description='Importera historik från Tipsextra CSV och fyll tabeller')
    parser.add_argument('csvfile', help='Sökväg till CSV-filen som ska importeras')
    parser.add_argument('--max', type=int, default=None, help='Max antal omgångar att importera (valfritt)')
    parser.add_argument('--replace', action='store_true', help='Om satt så ersätts befintliga matchrader för varje omgång innan insert')
    args = parser.parse_args()

    import_file(args.csvfile, max_omgangar=args.max, replace=args.replace)


if __name__ == '__main__':
    main()
