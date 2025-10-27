import os
import argparse
from itertools import combinations

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


def sanitize_rad(rad: str) -> str:
    return "".join(c for c in str(rad).upper() if c in ("1", "X", "2"))


def generate_candidates(base: str, max_dist: int):
    n = len(base)
    res = {base}
    if max_dist <= 0:
        return list(res)
    alt = {"1": ("X", "2"), "X": ("1", "2"), "2": ("1", "X")}
    for k in range(1, max_dist + 1):
        for idxs in combinations(range(n), k):
            arr = list(base)

            def dfs(p: int):
                if p == len(idxs):
                    res.add("".join(arr))
                    return
                i = idxs[p]
                orig = arr[i]
                for a in alt[orig]:
                    arr[i] = a
                    dfs(p + 1)
                arr[i] = orig

            dfs(0)
    return list(res)


def build_all_historik(min_matches: int = 13):
    with get_conn() as conn, conn.cursor() as cur:
        # Hitta kompletta omgångar (radlängd == antal matcher, och >= min_matches)
        cur.execute(
            """
            SELECT o.omgang_id,
                   regexp_replace(upper(k.rad), '[^12X]', '', 'g') AS srad,
                   COUNT(m.matchnummer) AS match_count,
                   o.oddset_radsumma_max,
                   o.oddset_radsumma_min,
                   o.svenska_folket_radsumma_max,
                   o.svenska_folket_radsumma_min
            FROM omgang o
            JOIN kombinationer k ON k.kombinations_id = o.correct
            JOIN omgangsmatch m ON m.omgang_id = o.omgang_id
            GROUP BY o.omgang_id, k.rad,
                     o.oddset_radsumma_max, o.oddset_radsumma_min,
                     o.svenska_folket_radsumma_max, o.svenska_folket_radsumma_min
            HAVING COUNT(m.matchnummer) = length(regexp_replace(k.rad, '[^12Xx]', '', 'g'))
               AND COUNT(m.matchnummer) >= %s
            ORDER BY o.omgang_id
            """,
            (min_matches,),
        )
        omg = cur.fetchall()
        print(f"Hittade {len(omg)} kompletta omgångar (>= {min_matches} matcher)")

        for (
            omgang_id,
            base_rad,
            match_count,
            o_radsum_max,
            o_radsum_min,
            p_radsum_max,
            p_radsum_min,
        ) in omg:
            max_dist = max(0, match_count - min_matches)
            candidates = generate_candidates(base_rad, max_dist)
            if not candidates:
                continue

            # Slå upp kombinations_id för kandidaterna
            cur.execute(
                "SELECT kombinations_id, rad FROM kombinationer WHERE rad = ANY(%s)",
                (candidates,),
            )
            kid_map = {row[1]: row[0] for row in cur.fetchall()}
            if not kid_map:
                continue

            # Hämta matchdata för omgången
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
            if not rows:
                continue

            mm = []
            for (mn, o1, ox, o2, p1, px, p2, op1, opx, op2, sp1, spx, sp2, orank, prank) in rows:
                mm.append(
                    {
                        "o_odds": {"1": o1, "X": ox, "2": o2},
                        "p_odds": {"1": p1, "X": px, "2": p2},
                        "o_pct": {"1": op1 or 0.0, "X": opx or 0.0, "2": op2 or 0.0},
                        "p_pct": {"1": sp1 or 0.0, "X": spx or 0.0, "2": sp2 or 0.0},
                        "o_rank": int(orank) if orank is not None else None,
                        "p_rank": int(prank) if prank is not None else None,
                    }
                )

            # Förbered kolumnordning
            base_cols = [
                "omgang_id",
                "rad",
                "correct",
                "oddset_radsumma",
                "svenska_folket_radsumma",
                "oddset_radsumma_max",
                "oddset_radsumma_min",
                "svenska_folket_radsumma_max",
                "svenska_folket_radsumma_min",
                "oddset_right_count",
                "oddset_even_count",
                "oddset_wrong_count",
                "people_right_count",
                "people_even_count",
                "people_wrong_count",
                "oddset_right_points",
                "oddset_even_points",
                "oddset_wrong_points",
                "people_right_points",
                "people_even_points",
                "people_wrong_points",
                "oddset_group1_right_count",
                "oddset_group2_right_count",
                "oddset_group3_right_count",
                "oddset_group1_even_count",
                "oddset_group2_even_count",
                "oddset_group3_even_count",
                "oddset_group1_wrong_count",
                "oddset_group2_wrong_count",
                "oddset_group3_wrong_count",
                "people_group1_right_count",
                "people_group2_right_count",
                "people_group3_right_count",
                "people_group1_even_count",
                "people_group2_even_count",
                "people_group3_even_count",
                "people_group1_wrong_count",
                "people_group2_wrong_count",
                "people_group3_wrong_count",
            ]
            rank_cols = []
            for r in range(1, 14):
                rank_cols += [
                    f"rank{r}_oddset_right",
                    f"rank{r}_oddset_even",
                    f"rank{r}_oddset_wrong",
                    f"rank{r}_people_right",
                    f"rank{r}_people_even",
                    f"rank{r}_people_wrong",
                ]
            cols = base_cols + rank_cols

            values = []
            for s in candidates:
                srad = sanitize_rad(s)
                if len(srad) != len(mm):
                    continue
                # Map rad -> kombinations_id
                kid = kid_map.get(srad)
                if kid is None:
                    continue
                # antal rätt = antal positioner där kandidat == basrad
                correct_val = sum(1 for i, ch in enumerate(srad) if ch == base_rad[i])

                o_sum = 0.0
                p_sum = 0.0
                oc_r = oc_e = oc_w = 0
                pc_r = pc_e = pc_w = 0
                og = {1: {"R": 0, "E": 0, "W": 0}, 2: {"R": 0, "E": 0, "W": 0}, 3: {"R": 0, "E": 0, "W": 0}}
                pg = {1: {"R": 0, "E": 0, "W": 0}, 2: {"R": 0, "E": 0, "W": 0}, 3: {"R": 0, "E": 0, "W": 0}}
                op_r = op_e = op_w = 0
                pp_r = pp_e = pp_w = 0
                flags = {name: 0 for name in rank_cols}

                for pos, ch in enumerate(srad):
                    data = mm[pos]
                    o_sum += float(data["o_pct"].get(ch, 0.0))
                    p_sum += float(data["p_pct"].get(ch, 0.0))

                    ovals = [v for v in data["o_odds"].values() if v is not None]
                    pvals = [v for v in data["p_odds"].values() if v is not None]
                    # Oddset
                    if ovals:
                        omin = min(ovals)
                        omax = max(ovals)
                        ochoice = data["o_odds"].get(ch)
                        if ochoice == omin and ochoice != omax:
                            ocls = "R"
                            oc_r += 1
                        elif ochoice == omax and ochoice != omin:
                            ocls = "W"
                            oc_w += 1
                        else:
                            ocls = "E"
                            oc_e += 1
                        orank = data["o_rank"]
                        if isinstance(orank, int) and 1 <= orank <= 13:
                            flags[f"rank{orank}_oddset_{'right' if ocls=='R' else 'wrong' if ocls=='W' else 'even'}"] = 1
                            pts = 14 - orank
                            if ocls == "R":
                                op_r += pts
                            elif ocls == "W":
                                op_w += pts
                            else:
                                op_e += pts
                            if 1 <= orank <= 3:
                                og[1][ocls] += 1
                            elif 4 <= orank <= 8:
                                og[2][ocls] += 1
                            else:
                                og[3][ocls] += 1
                    # People
                    if pvals:
                        pmin = min(pvals)
                        pmax = max(pvals)
                        pchoice = data["p_odds"].get(ch)
                        if pchoice == pmin and pchoice != pmax:
                            pcls = "R"
                            pc_r += 1
                        elif pchoice == pmax and pchoice != pmin:
                            pcls = "W"
                            pc_w += 1
                        else:
                            pcls = "E"
                            pc_e += 1
                        prank = data["p_rank"]
                        if isinstance(prank, int) and 1 <= prank <= 13:
                            flags[f"rank{prank}_people_{'right' if pcls=='R' else 'wrong' if pcls=='W' else 'even'}"] = 1
                            pts = 14 - prank
                            if pcls == "R":
                                pp_r += pts
                            elif pcls == "W":
                                pp_w += pts
                            else:
                                pp_e += pts
                            if 1 <= prank <= 3:
                                pg[1][pcls] += 1
                            elif 4 <= prank <= 8:
                                pg[2][pcls] += 1
                            else:
                                pg[3][pcls] += 1

                row = [
                    omgang_id,
                    kid,
                    correct_val,
                    o_sum,
                    p_sum,
                    o_radsum_max,
                    o_radsum_min,
                    p_radsum_max,
                    p_radsum_min,
                    oc_r,
                    oc_e,
                    oc_w,
                    pc_r,
                    pc_e,
                    pc_w,
                    op_r,
                    op_e,
                    op_w,
                    pp_r,
                    pp_e,
                    pp_w,
                    og[1]["R"], og[2]["R"], og[3]["R"],
                    og[1]["E"], og[2]["E"], og[3]["E"],
                    og[1]["W"], og[2]["W"], og[3]["W"],
                    pg[1]["R"], pg[2]["R"], pg[3]["R"],
                    pg[1]["E"], pg[2]["E"], pg[3]["E"],
                    pg[1]["W"], pg[2]["W"], pg[3]["W"],
                ]
                for name in rank_cols:
                    row.append(flags[name])
                values.append(tuple(row))

            if values:
                insert_sql = f"INSERT INTO historik ({', '.join(cols)}) VALUES %s ON CONFLICT (omgang_id, rad) DO UPDATE SET " + \
                    ", ".join([f"{c}=EXCLUDED.{c}" for c in cols if c not in ("omgang_id", "rad")])
                execute_values(cur, insert_sql, values)
                conn.commit()
                print(f"Historik: omg {omgang_id} — {len(values)} kandidater")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bygg historik (träningsdata) för alla kompletta omgångar")
    ap.add_argument("--min-matches", type=int, default=13, help="Minsta antal matcher som måste vara kompletta (default 13)")
    args = ap.parse_args()
    build_all_historik(min_matches=args.min_matches)
