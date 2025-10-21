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
               oddset1, oddsetx, oddset2,
               svenska_folket_odds1, svenska_folket_oddsx, svenska_folket_odds2,
               oddset_procent1, oddset_procentx, oddset_procent2,
               svenska_folket_procent1, svenska_folket_procentx, svenska_folket_procent2,
               oddset_rank, people_rank
        FROM omgangsmatch
        WHERE omgang_id = %s
        ORDER BY matchnummer
        """,
        (omgang_id,),
    )
    rows = cur.fetchall()
    # Indexera 0-baserat efter matchnummer-1
    oddset_pct_map = {}
    svenska_pct_map = {}
    oddset_odds_map = {}
    people_odds_map = {}
    oddset_rank_map = {}
    people_rank_map = {}
    for (mn, o1_o, ox_o, o2_o, p1_o, px_o, p2_o, o1, ox, o2, s1, sx, s2, o_rank, p_rank) in rows:
        pos = int(mn) - 1
        oddset_pct_map[pos] = {
            "1": float(o1) if o1 is not None else 0.0,
            "X": float(ox) if ox is not None else 0.0,
            "2": float(o2) if o2 is not None else 0.0,
        }
        svenska_pct_map[pos] = {
            "1": float(s1) if s1 is not None else 0.0,
            "X": float(sx) if sx is not None else 0.0,
            "2": float(s2) if s2 is not None else 0.0,
        }
        oddset_odds_map[pos] = {
            "1": float(o1_o) if o1_o is not None else None,
            "X": float(ox_o) if ox_o is not None else None,
            "2": float(o2_o) if o2_o is not None else None,
        }
        people_odds_map[pos] = {
            "1": float(p1_o) if p1_o is not None else None,
            "X": float(px_o) if px_o is not None else None,
            "2": float(p2_o) if p2_o is not None else None,
        }
        oddset_rank_map[pos] = int(o_rank) if o_rank is not None else None
        people_rank_map[pos] = int(p_rank) if p_rank is not None else None
    return (
        oddset_pct_map,
        svenska_pct_map,
        len(rows),
        oddset_odds_map,
        people_odds_map,
        oddset_rank_map,
        people_rank_map,
    )


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
        (
            oddset_map,
            svenska_map,
            match_count,
            oddset_odds_map,
            people_odds_map,
            oddset_rank_map,
            people_rank_map,
        ) = fetch_match_data(cur, omgang_id)
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

            # Förbered kolumnlista för rank‑flaggor 1..13
            flag_cols = []
            for rnk in range(1, 14):
                flag_cols += [
                    f"rank{rnk}_oddset_right",
                    f"rank{rnk}_oddset_even",
                    f"rank{rnk}_oddset_wrong",
                    f"rank{rnk}_people_right",
                    f"rank{rnk}_people_even",
                    f"rank{rnk}_people_wrong",
                ]

            updates = []  # tuples matching (id, o_sum, s_sum, counts..., group_counts..., flags...)
            for cid, rad in rows:
                s = sanitize_rad(rad)
                odd_sum = 0.0
                sv_sum = 0.0
                # Count aggregations
                oc_right = oc_even = oc_wrong = 0
                pc_right = pc_even = pc_wrong = 0
                # Grouped count aggregations
                og1_r = og1_e = og1_w = 0
                og2_r = og2_e = og2_w = 0
                og3_r = og3_e = og3_w = 0
                pg1_r = pg1_e = pg1_w = 0
                pg2_r = pg2_e = pg2_w = 0
                pg3_r = pg3_e = pg3_w = 0
                flags = {col: 0 for col in flag_cols}
                # Summera över antalet matcher vi har data för
                for pos in range(match_count):
                    if pos < len(s):
                        ch = s[pos]
                        odd_sum += oddset_map.get(pos, {}).get(ch, 0.0)
                        sv_sum += svenska_map.get(pos, {}).get(ch, 0.0)
                        # Klassificering för rank (oddset & people) baserat på odds
                        o_odds = oddset_odds_map.get(pos, {})
                        p_odds = people_odds_map.get(pos, {})
                        o_choice = o_odds.get(ch)
                        p_choice = p_odds.get(ch)
                        # Beräkna min/max per källa, hantera None genom att ignorera
                        o_vals_all = [v for v in o_odds.values() if v is not None]
                        p_vals_all = [v for v in p_odds.values() if v is not None]
                        if o_vals_all and o_choice is not None:
                            o_min = min(o_vals_all)
                            o_max = max(o_vals_all)
                            if o_choice == o_min and o_choice != o_max:
                                o_cls = 'right'
                                oc_right += 1
                            elif o_choice == o_max and o_choice != o_min:
                                o_cls = 'wrong'
                                oc_wrong += 1
                            else:
                                o_cls = 'even'
                                oc_even += 1
                            o_rank = oddset_rank_map.get(pos)
                            if isinstance(o_rank, int) and 1 <= o_rank <= 13:
                                key = f"rank{o_rank}_oddset_{o_cls}"
                                flags[key] = 1
                                # Group bucket for oddset rank
                                if 1 <= o_rank <= 3:
                                    if o_cls == 'right': og1_r += 1
                                    elif o_cls == 'even': og1_e += 1
                                    else: og1_w += 1
                                elif 4 <= o_rank <= 8:
                                    if o_cls == 'right': og2_r += 1
                                    elif o_cls == 'even': og2_e += 1
                                    else: og2_w += 1
                                elif 9 <= o_rank <= 13:
                                    if o_cls == 'right': og3_r += 1
                                    elif o_cls == 'even': og3_e += 1
                                    else: og3_w += 1
                        if p_vals_all and p_choice is not None:
                            p_min = min(p_vals_all)
                            p_max = max(p_vals_all)
                            if p_choice == p_min and p_choice != p_max:
                                p_cls = 'right'
                                pc_right += 1
                            elif p_choice == p_max and p_choice != p_min:
                                p_cls = 'wrong'
                                pc_wrong += 1
                            else:
                                p_cls = 'even'
                                pc_even += 1
                            p_rank = people_rank_map.get(pos)
                            if isinstance(p_rank, int) and 1 <= p_rank <= 13:
                                key = f"rank{p_rank}_people_{p_cls}"
                                flags[key] = 1
                                # Group bucket for people rank
                                if 1 <= p_rank <= 3:
                                    if p_cls == 'right': pg1_r += 1
                                    elif p_cls == 'even': pg1_e += 1
                                    else: pg1_w += 1
                                elif 4 <= p_rank <= 8:
                                    if p_cls == 'right': pg2_r += 1
                                    elif p_cls == 'even': pg2_e += 1
                                    else: pg2_w += 1
                                elif 9 <= p_rank <= 13:
                                    if p_cls == 'right': pg3_r += 1
                                    elif p_cls == 'even': pg3_e += 1
                                    else: pg3_w += 1
                updates.append((cid, odd_sum, sv_sum, oc_right, oc_even, oc_wrong, pc_right, pc_even, pc_wrong,
                                og1_r, og1_e, og1_w, og2_r, og2_e, og2_w, og3_r, og3_e, og3_w,
                                pg1_r, pg1_e, pg1_w, pg2_r, pg2_e, pg2_w, pg3_r, pg3_e, pg3_w,
                                *[flags[c] for c in flag_cols]))
                last_id = cid

            # Bulk-uppdatera via VALUES‑tabell med alla rank‑flaggor
            values_cols = [
                "id", "o", "s",
                "oc_r", "oc_e", "oc_w", "pc_r", "pc_e", "pc_w",
                "og1_r", "og1_e", "og1_w", "og2_r", "og2_e", "og2_w", "og3_r", "og3_e", "og3_w",
                "pg1_r", "pg1_e", "pg1_w", "pg2_r", "pg2_e", "pg2_w", "pg3_r", "pg3_e", "pg3_w"
            ] + flag_cols
            values_cols_sql = ", ".join(values_cols)
            set_cols_sql = ", ".join([
                "oddset_radsumma = v.o",
                "svenska_folket_radsumma = v.s",
                "oddset_right_count = v.oc_r",
                "oddset_even_count = v.oc_e",
                "oddset_wrong_count = v.oc_w",
                "people_right_count = v.pc_r",
                "people_even_count = v.pc_e",
                "people_wrong_count = v.pc_w",
                "oddset_group1_right_count = v.og1_r",
                "oddset_group1_even_count = v.og1_e",
                "oddset_group1_wrong_count = v.og1_w",
                "oddset_group2_right_count = v.og2_r",
                "oddset_group2_even_count = v.og2_e",
                "oddset_group2_wrong_count = v.og2_w",
                "oddset_group3_right_count = v.og3_r",
                "oddset_group3_even_count = v.og3_e",
                "oddset_group3_wrong_count = v.og3_w",
                "people_group1_right_count = v.pg1_r",
                "people_group1_even_count = v.pg1_e",
                "people_group1_wrong_count = v.pg1_w",
                "people_group2_right_count = v.pg2_r",
                "people_group2_even_count = v.pg2_e",
                "people_group2_wrong_count = v.pg2_w",
                "people_group3_right_count = v.pg3_r",
                "people_group3_even_count = v.pg3_e",
                "people_group3_wrong_count = v.pg3_w",
            ] + [f"{col} = v.{col}" for col in flag_cols])
            update_sql = f"""
                UPDATE kommande AS c
                SET {set_cols_sql}
                FROM (VALUES %s) AS v({values_cols_sql})
                WHERE c.id = v.id
            """
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
                svenska_folket_radsumma = sums.sv_sum,
                oddset_right_count = sums.oc_right,
                oddset_even_count = sums.oc_even,
                oddset_wrong_count = sums.oc_wrong,
                people_right_count = sums.pc_right,
                people_even_count = sums.pc_even,
                people_wrong_count = sums.pc_wrong,
                oddset_group1_right_count = sums.og1_r,
                oddset_group1_even_count = sums.og1_e,
                oddset_group1_wrong_count = sums.og1_w,
                oddset_group2_right_count = sums.og2_r,
                oddset_group2_even_count = sums.og2_e,
                oddset_group2_wrong_count = sums.og2_w,
                oddset_group3_right_count = sums.og3_r,
                oddset_group3_even_count = sums.og3_e,
                oddset_group3_wrong_count = sums.og3_w,
                people_group1_right_count = sums.pg1_r,
                people_group1_even_count = sums.pg1_e,
                people_group1_wrong_count = sums.pg1_w,
                people_group2_right_count = sums.pg2_r,
                people_group2_even_count = sums.pg2_e,
                people_group2_wrong_count = sums.pg2_w,
                people_group3_right_count = sums.pg3_r,
                people_group3_even_count = sums.pg3_e,
                people_group3_wrong_count = sums.pg3_w,
                rank1_oddset_right = sums.rank1_oddset_right,
                rank1_oddset_even = sums.rank1_oddset_even,
                rank1_oddset_wrong = sums.rank1_oddset_wrong,
                rank1_people_right = sums.rank1_people_right,
                rank1_people_even = sums.rank1_people_even,
                rank1_people_wrong = sums.rank1_people_wrong,
                rank2_oddset_right = sums.rank2_oddset_right,
                rank2_oddset_even = sums.rank2_oddset_even,
                rank2_oddset_wrong = sums.rank2_oddset_wrong,
                rank2_people_right = sums.rank2_people_right,
                rank2_people_even = sums.rank2_people_even,
                rank2_people_wrong = sums.rank2_people_wrong,
                rank3_oddset_right = sums.rank3_oddset_right,
                rank3_oddset_even = sums.rank3_oddset_even,
                rank3_oddset_wrong = sums.rank3_oddset_wrong,
                rank3_people_right = sums.rank3_people_right,
                rank3_people_even = sums.rank3_people_even,
                rank3_people_wrong = sums.rank3_people_wrong,
                rank4_oddset_right = sums.rank4_oddset_right,
                rank4_oddset_even = sums.rank4_oddset_even,
                rank4_oddset_wrong = sums.rank4_oddset_wrong,
                rank4_people_right = sums.rank4_people_right,
                rank4_people_even = sums.rank4_people_even,
                rank4_people_wrong = sums.rank4_people_wrong,
                rank5_oddset_right = sums.rank5_oddset_right,
                rank5_oddset_even = sums.rank5_oddset_even,
                rank5_oddset_wrong = sums.rank5_oddset_wrong,
                rank5_people_right = sums.rank5_people_right,
                rank5_people_even = sums.rank5_people_even,
                rank5_people_wrong = sums.rank5_people_wrong,
                rank6_oddset_right = sums.rank6_oddset_right,
                rank6_oddset_even = sums.rank6_oddset_even,
                rank6_oddset_wrong = sums.rank6_oddset_wrong,
                rank6_people_right = sums.rank6_people_right,
                rank6_people_even = sums.rank6_people_even,
                rank6_people_wrong = sums.rank6_people_wrong,
                rank7_oddset_right = sums.rank7_oddset_right,
                rank7_oddset_even = sums.rank7_oddset_even,
                rank7_oddset_wrong = sums.rank7_oddset_wrong,
                rank7_people_right = sums.rank7_people_right,
                rank7_people_even = sums.rank7_people_even,
                rank7_people_wrong = sums.rank7_people_wrong,
                rank8_oddset_right = sums.rank8_oddset_right,
                rank8_oddset_even = sums.rank8_oddset_even,
                rank8_oddset_wrong = sums.rank8_oddset_wrong,
                rank8_people_right = sums.rank8_people_right,
                rank8_people_even = sums.rank8_people_even,
                rank8_people_wrong = sums.rank8_people_wrong,
                rank9_oddset_right = sums.rank9_oddset_right,
                rank9_oddset_even = sums.rank9_oddset_even,
                rank9_oddset_wrong = sums.rank9_oddset_wrong,
                rank9_people_right = sums.rank9_people_right,
                rank9_people_even = sums.rank9_people_even,
                rank9_people_wrong = sums.rank9_people_wrong,
                rank10_oddset_right = sums.rank10_oddset_right,
                rank10_oddset_even = sums.rank10_oddset_even,
                rank10_oddset_wrong = sums.rank10_oddset_wrong,
                rank10_people_right = sums.rank10_people_right,
                rank10_people_even = sums.rank10_people_even,
                rank10_people_wrong = sums.rank10_people_wrong,
                rank11_oddset_right = sums.rank11_oddset_right,
                rank11_oddset_even = sums.rank11_oddset_even,
                rank11_oddset_wrong = sums.rank11_oddset_wrong,
                rank11_people_right = sums.rank11_people_right,
                rank11_people_even = sums.rank11_people_even,
                rank11_people_wrong = sums.rank11_people_wrong,
                rank12_oddset_right = sums.rank12_oddset_right,
                rank12_oddset_even = sums.rank12_oddset_even,
                rank12_oddset_wrong = sums.rank12_oddset_wrong,
                rank12_people_right = sums.rank12_people_right,
                rank12_people_even = sums.rank12_people_even,
                rank12_people_wrong = sums.rank12_people_wrong,
                rank13_oddset_right = sums.rank13_oddset_right,
                rank13_oddset_even = sums.rank13_oddset_even,
                rank13_oddset_wrong = sums.rank13_oddset_wrong,
                rank13_people_right = sums.rank13_people_right,
                rank13_people_even = sums.rank13_people_even,
                rank13_people_wrong = sums.rank13_people_wrong
            FROM (
                WITH mm AS (
                    SELECT matchnummer,
                           oddset1, oddsetx, oddset2,
                           svenska_folket_odds1, svenska_folket_oddsx, svenska_folket_odds2,
                           oddset_procent1, oddset_procentx, oddset_procent2,
                           svenska_folket_procent1, svenska_folket_procentx, svenska_folket_procent2,
                           oddset_rank, people_rank
                    FROM omgangsmatch
                    WHERE omgang_id = %s
                )
                SELECT c2.id,
                       SUM(CASE ch
                               WHEN '1' THEN COALESCE(mm.oddset_procent1, 0)
                               WHEN 'X' THEN COALESCE(mm.oddset_procentx, 0)
                               WHEN '2' THEN COALESCE(mm.oddset_procent2, 0)
                               ELSE 0 END) AS odd_sum,
                       SUM(CASE ch
                               WHEN '1' THEN COALESCE(mm.svenska_folket_procent1, 0)
                               WHEN 'X' THEN COALESCE(mm.svenska_folket_procentx, 0)
                               WHEN '2' THEN COALESCE(mm.svenska_folket_procent2, 0)
                               ELSE 0 END) AS sv_sum,
                       SUM(CASE WHEN cls_o = 'R' THEN 1 ELSE 0 END) AS oc_right,
                       SUM(CASE WHEN cls_o = 'E' THEN 1 ELSE 0 END) AS oc_even,
                       SUM(CASE WHEN cls_o = 'W' THEN 1 ELSE 0 END) AS oc_wrong,
                       SUM(CASE WHEN cls_p = 'R' THEN 1 ELSE 0 END) AS pc_right,
                       SUM(CASE WHEN cls_p = 'E' THEN 1 ELSE 0 END) AS pc_even,
                       SUM(CASE WHEN cls_p = 'W' THEN 1 ELSE 0 END) AS pc_wrong,
                       SUM(CASE WHEN mm.oddset_rank BETWEEN 1 AND 3 AND cls_o = 'R' THEN 1 ELSE 0 END) AS og1_r,
                       SUM(CASE WHEN mm.oddset_rank BETWEEN 1 AND 3 AND cls_o = 'E' THEN 1 ELSE 0 END) AS og1_e,
                       SUM(CASE WHEN mm.oddset_rank BETWEEN 1 AND 3 AND cls_o = 'W' THEN 1 ELSE 0 END) AS og1_w,
                       SUM(CASE WHEN mm.oddset_rank BETWEEN 4 AND 8 AND cls_o = 'R' THEN 1 ELSE 0 END) AS og2_r,
                       SUM(CASE WHEN mm.oddset_rank BETWEEN 4 AND 8 AND cls_o = 'E' THEN 1 ELSE 0 END) AS og2_e,
                       SUM(CASE WHEN mm.oddset_rank BETWEEN 4 AND 8 AND cls_o = 'W' THEN 1 ELSE 0 END) AS og2_w,
                       SUM(CASE WHEN mm.oddset_rank BETWEEN 9 AND 13 AND cls_o = 'R' THEN 1 ELSE 0 END) AS og3_r,
                       SUM(CASE WHEN mm.oddset_rank BETWEEN 9 AND 13 AND cls_o = 'E' THEN 1 ELSE 0 END) AS og3_e,
                       SUM(CASE WHEN mm.oddset_rank BETWEEN 9 AND 13 AND cls_o = 'W' THEN 1 ELSE 0 END) AS og3_w,
                       SUM(CASE WHEN mm.people_rank BETWEEN 1 AND 3 AND cls_p = 'R' THEN 1 ELSE 0 END) AS pg1_r,
                       SUM(CASE WHEN mm.people_rank BETWEEN 1 AND 3 AND cls_p = 'E' THEN 1 ELSE 0 END) AS pg1_e,
                       SUM(CASE WHEN mm.people_rank BETWEEN 1 AND 3 AND cls_p = 'W' THEN 1 ELSE 0 END) AS pg1_w,
                       SUM(CASE WHEN mm.people_rank BETWEEN 4 AND 8 AND cls_p = 'R' THEN 1 ELSE 0 END) AS pg2_r,
                       SUM(CASE WHEN mm.people_rank BETWEEN 4 AND 8 AND cls_p = 'E' THEN 1 ELSE 0 END) AS pg2_e,
                       SUM(CASE WHEN mm.people_rank BETWEEN 4 AND 8 AND cls_p = 'W' THEN 1 ELSE 0 END) AS pg2_w,
                       SUM(CASE WHEN mm.people_rank BETWEEN 9 AND 13 AND cls_p = 'R' THEN 1 ELSE 0 END) AS pg3_r,
                       SUM(CASE WHEN mm.people_rank BETWEEN 9 AND 13 AND cls_p = 'E' THEN 1 ELSE 0 END) AS pg3_e,
                       SUM(CASE WHEN mm.people_rank BETWEEN 9 AND 13 AND cls_p = 'W' THEN 1 ELSE 0 END) AS pg3_w,
                       SUM(CASE WHEN mm.oddset_rank = 1 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank1_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 1 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank1_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 1 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank1_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 1 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank1_people_right,
                       SUM(CASE WHEN mm.people_rank = 1 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank1_people_even,
                       SUM(CASE WHEN mm.people_rank = 1 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank1_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 2 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank2_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 2 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank2_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 2 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank2_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 2 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank2_people_right,
                       SUM(CASE WHEN mm.people_rank = 2 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank2_people_even,
                       SUM(CASE WHEN mm.people_rank = 2 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank2_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 3 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank3_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 3 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank3_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 3 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank3_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 3 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank3_people_right,
                       SUM(CASE WHEN mm.people_rank = 3 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank3_people_even,
                       SUM(CASE WHEN mm.people_rank = 3 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank3_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 4 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank4_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 4 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank4_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 4 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank4_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 4 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank4_people_right,
                       SUM(CASE WHEN mm.people_rank = 4 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank4_people_even,
                       SUM(CASE WHEN mm.people_rank = 4 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank4_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 5 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank5_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 5 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank5_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 5 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank5_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 5 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank5_people_right,
                       SUM(CASE WHEN mm.people_rank = 5 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank5_people_even,
                       SUM(CASE WHEN mm.people_rank = 5 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank5_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 6 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank6_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 6 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank6_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 6 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank6_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 6 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank6_people_right,
                       SUM(CASE WHEN mm.people_rank = 6 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank6_people_even,
                       SUM(CASE WHEN mm.people_rank = 6 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank6_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 7 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank7_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 7 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank7_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 7 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank7_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 7 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank7_people_right,
                       SUM(CASE WHEN mm.people_rank = 7 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank7_people_even,
                       SUM(CASE WHEN mm.people_rank = 7 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank7_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 8 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank8_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 8 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank8_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 8 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank8_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 8 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank8_people_right,
                       SUM(CASE WHEN mm.people_rank = 8 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank8_people_even,
                       SUM(CASE WHEN mm.people_rank = 8 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank8_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 9 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank9_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 9 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank9_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 9 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank9_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 9 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank9_people_right,
                       SUM(CASE WHEN mm.people_rank = 9 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank9_people_even,
                       SUM(CASE WHEN mm.people_rank = 9 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank9_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 10 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank10_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 10 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank10_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 10 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank10_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 10 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank10_people_right,
                       SUM(CASE WHEN mm.people_rank = 10 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank10_people_even,
                       SUM(CASE WHEN mm.people_rank = 10 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank10_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 11 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank11_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 11 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank11_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 11 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank11_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 11 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank11_people_right,
                       SUM(CASE WHEN mm.people_rank = 11 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank11_people_even,
                       SUM(CASE WHEN mm.people_rank = 11 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank11_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 12 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank12_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 12 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank12_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 12 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank12_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 12 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank12_people_right,
                       SUM(CASE WHEN mm.people_rank = 12 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank12_people_even,
                       SUM(CASE WHEN mm.people_rank = 12 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank12_people_wrong,
                       SUM(CASE WHEN mm.oddset_rank = 13 AND cls_o = 'R' THEN 1 ELSE 0 END) AS rank13_oddset_right,
                       SUM(CASE WHEN mm.oddset_rank = 13 AND cls_o = 'E' THEN 1 ELSE 0 END) AS rank13_oddset_even,
                       SUM(CASE WHEN mm.oddset_rank = 13 AND cls_o = 'W' THEN 1 ELSE 0 END) AS rank13_oddset_wrong,
                       SUM(CASE WHEN mm.people_rank = 13 AND cls_p = 'R' THEN 1 ELSE 0 END) AS rank13_people_right,
                       SUM(CASE WHEN mm.people_rank = 13 AND cls_p = 'E' THEN 1 ELSE 0 END) AS rank13_people_even,
                       SUM(CASE WHEN mm.people_rank = 13 AND cls_p = 'W' THEN 1 ELSE 0 END) AS rank13_people_wrong
                FROM kommande c2
                JOIN kombinationer k ON k.kombinations_id = c2.rad
                CROSS JOIN LATERAL (
                    SELECT regexp_replace(upper(k.rad), '[^12X]', '', 'g') AS srad
                ) s2
                JOIN mm ON TRUE
                CROSS JOIN LATERAL (
                    SELECT SUBSTRING(s2.srad FROM mm.matchnummer FOR 1) AS ch,
                           CASE
                               WHEN o_choice IS NULL THEN 'N'
                               WHEN o_choice = o_min AND o_choice <> o_max THEN 'R'
                               WHEN o_choice = o_max AND o_choice <> o_min THEN 'W'
                               ELSE 'E'
                           END AS cls_o,
                           CASE
                               WHEN p_choice IS NULL THEN 'N'
                               WHEN p_choice = p_min AND p_choice <> p_max THEN 'R'
                               WHEN p_choice = p_max AND p_choice <> p_min THEN 'W'
                               ELSE 'E'
                           END AS cls_p
                    FROM (
                        SELECT
                            CASE SUBSTRING(s2.srad FROM mm.matchnummer FOR 1)
                                WHEN '1' THEN mm.oddset1
                                WHEN 'X' THEN mm.oddsetx
                                WHEN '2' THEN mm.oddset2
                                ELSE NULL END AS o_choice,
                            LEAST(mm.oddset1, mm.oddsetx, mm.oddset2) AS o_min,
                            GREATEST(mm.oddset1, mm.oddsetx, mm.oddset2) AS o_max,
                            CASE SUBSTRING(s2.srad FROM mm.matchnummer FOR 1)
                                WHEN '1' THEN mm.svenska_folket_odds1
                                WHEN 'X' THEN mm.svenska_folket_oddsx
                                WHEN '2' THEN mm.svenska_folket_odds2
                                ELSE NULL END AS p_choice,
                            LEAST(mm.svenska_folket_odds1, mm.svenska_folket_oddsx, mm.svenska_folket_odds2) AS p_min,
                            GREATEST(mm.svenska_folket_odds1, mm.svenska_folket_oddsx, mm.svenska_folket_odds2) AS p_max
                    ) t
                ) cls
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
