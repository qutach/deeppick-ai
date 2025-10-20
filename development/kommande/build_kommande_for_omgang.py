import os
import argparse
import psycopg2
from psycopg2.extras import execute_values


def get_conn():
    return psycopg2.connect(
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
    )


def fetch_match_data(cur, omgang_id: int):
    cur.execute(
        """
        SELECT matchnummer,
               oddset_procent1, oddset_procentx, oddset_procent2,
               svenska_folket_procent1, svenska_folket_procentx, svenska_folket_procent2
        FROM omgangsmatch
        WHERE omgang_id = %s
        ORDER BY matchnummer
        """,
        (omgang_id,),
    )
    rows = cur.fetchall()
    # Indexera 0-baserat efter matchnummer-1
    oddset_map = {}
    svenska_map = {}
    for (mn, o1, ox, o2, s1, sx, s2) in rows:
        pos = int(mn) - 1
        oddset_map[pos] = {
            "1": float(o1) if o1 is not None else 0.0,
            "X": float(ox) if ox is not None else 0.0,
            "2": float(o2) if o2 is not None else 0.0,
        }
        svenska_map[pos] = {
            "1": float(s1) if s1 is not None else 0.0,
            "X": float(sx) if sx is not None else 0.0,
            "2": float(s2) if s2 is not None else 0.0,
        }
    return oddset_map, svenska_map, len(rows)


def sanitize_rad(rad: str) -> str:
    if rad is None:
        return ""
    return "".join(c.upper() for c in str(rad) if c.upper() in ("1", "X", "2"))


def ensure_kommande_rows(conn, omgang_id: int, batch_size: int):
    with conn.cursor() as cur:
        # Räkna antalet kombinationer för progress
        cur.execute("SELECT COUNT(*) FROM kombinationer")
        total = cur.fetchone()[0]
        inserted_total = 0

        offset = 0
        while offset < total:
            cur.execute(
                "SELECT kombinations_id FROM kombinationer ORDER BY kombinations_id OFFSET %s LIMIT %s",
                (offset, batch_size),
            )
            ids = [row[0] for row in cur.fetchall()]
            if not ids:
                break

            values = [(omgang_id, kid) for kid in ids]
            execute_values(
                cur,
                "INSERT INTO kommande (omgang_id, rad) VALUES %s ON CONFLICT (omgang_id, rad) DO NOTHING",
                values,
            )
            conn.commit()

            inserted_total += len(values)
            print(f"Kommande insert batch: {inserted_total}/{total}")
            offset += batch_size


def ensure_kommande_rows_fast(conn, omgang_id: int):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO kommande (omgang_id, rad)
            SELECT %s, k.kombinations_id
            FROM kombinationer k
            ON CONFLICT (omgang_id, rad) DO NOTHING
            """,
            (omgang_id,),
        )
        conn.commit()
        print("Kommande insert: klart (set-based)")


def update_kommande_sums(conn, omgang_id: int, batch_size: int):
    with conn.cursor() as cur:
        oddset_map, svenska_map, match_count = fetch_match_data(cur, omgang_id)
        if match_count == 0:
            print("Inga matcher hittades för denna omgång — hoppar uppdatering av summor.")
            return

        # Iterera i id-ordning, undvik OFFSET för stor tabell
        last_id = 0
        processed = 0
        while True:
            cur.execute(
                """
                SELECT c.id, k.rad
                FROM kommande c
                JOIN kombinationer k ON k.kombinations_id = c.rad
                WHERE c.omgang_id = %s
                  AND (c.oddset_radsumma IS NULL OR c.svenska_folket_radsumma IS NULL)
                  AND c.id > %s
                ORDER BY c.id
                LIMIT %s
                """,
                (omgang_id, last_id, batch_size),
            )
            rows = cur.fetchall()
            if not rows:
                break

            updates = []  # (id, oddset_sum, svenska_sum)
            for cid, rad in rows:
                s = sanitize_rad(rad)
                odd_sum = 0.0
                sv_sum = 0.0
                # Summera över antalet matcher vi har data för
                for pos in range(match_count):
                    if pos < len(s):
                        ch = s[pos]
                        odd_sum += oddset_map.get(pos, {}).get(ch, 0.0)
                        sv_sum += svenska_map.get(pos, {}).get(ch, 0.0)
                updates.append((cid, odd_sum, sv_sum))
                last_id = cid

            # Bulk-uppdatera via VALUES‑tabell
            update_sql = (
                """
                UPDATE kommande AS c
                SET oddset_radsumma = v.o,
                    svenska_folket_radsumma = v.s
                FROM (VALUES %s) AS v(id, o, s)
                WHERE c.id = v.id
                """
            )
            execute_values(cur, update_sql, updates)
            conn.commit()

            processed += len(updates)
            print(f"Kommande update batch: {processed} uppdaterade")


def update_kommande_sums_fast(conn, omgang_id: int):
    """Set-baserad uppdatering av summor för en hel omgång i en SQL."""
    with conn.cursor() as cur:
        sql = (
            """
            UPDATE kommande AS c
            SET oddset_radsumma = sums.odd_sum,
                svenska_folket_radsumma = sums.sv_sum
            FROM (
                WITH mm AS (
                    SELECT matchnummer,
                           oddset_procent1, oddset_procentx, oddset_procent2,
                           svenska_folket_procent1, svenska_folket_procentx, svenska_folket_procent2
                    FROM omgangsmatch
                    WHERE omgang_id = %s
                ),
                mc AS (
                    SELECT COUNT(*) AS match_count FROM mm
                )
                SELECT c2.id,
                       SUM(CASE SUBSTRING(s2.srad, gs.pos, 1)
                               WHEN '1' THEN COALESCE(mm.oddset_procent1, 0)
                               WHEN 'X' THEN COALESCE(mm.oddset_procentx, 0)
                               WHEN '2' THEN COALESCE(mm.oddset_procent2, 0)
                               ELSE 0 END) AS odd_sum,
                       SUM(CASE SUBSTRING(s2.srad, gs.pos, 1)
                               WHEN '1' THEN COALESCE(mm.svenska_folket_procent1, 0)
                               WHEN 'X' THEN COALESCE(mm.svenska_folket_procentx, 0)
                               WHEN '2' THEN COALESCE(mm.svenska_folket_procent2, 0)
                               ELSE 0 END) AS sv_sum
                FROM kommande c2
                JOIN kombinationer k ON k.kombinations_id = c2.rad
                CROSS JOIN LATERAL (
                    SELECT regexp_replace(upper(k.rad), '[^12X]', '', 'g') AS srad
                ) s2
                CROSS JOIN mc
                JOIN generate_series(1, mc.match_count) AS gs(pos) ON TRUE
                LEFT JOIN mm ON mm.matchnummer = gs.pos
                WHERE c2.omgang_id = %s
                GROUP BY c2.id
            ) AS sums
            WHERE c.id = sums.id AND c.omgang_id = %s
            """
        )
        cur.execute(sql, (omgang_id, omgang_id, omgang_id))
        conn.commit()
        print("Kommande update: klart (set-based)")


def main():
    ap = argparse.ArgumentParser(description="Skapa och uppdatera kommande-rader för en given omgång")
    ap.add_argument("--omgang-id", type=int, required=True, help="Omgang ID")
    ap.add_argument("--batch-size", type=int, default=100_000, help="Batch-storlek (default 100000)")
    ap.add_argument("--insert-only", action="store_true", help="Endast skapa rad-par, räkna inte summor")
    ap.add_argument("--fast", action="store_true", help="Använd set-baserad (snabb) insättning och uppdatering")
    args = ap.parse_args()

    with get_conn() as conn:
        if args.fast:
            ensure_kommande_rows_fast(conn, args.omgang_id)
        else:
            ensure_kommande_rows(conn, args.omgang_id, args.batch_size)
        if not args.insert_only:
            if args.fast:
                update_kommande_sums_fast(conn, args.omgang_id)
            else:
                update_kommande_sums(conn, args.omgang_id, args.batch_size)


if __name__ == "__main__":
    main()
