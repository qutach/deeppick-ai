import os
import argparse
import psycopg2


def get_conn():
    return psycopg2.connect(
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
    )


def get_match_count(cur, omgang_id: int) -> int:
    cur.execute("SELECT COUNT(*) FROM omgangsmatch WHERE omgang_id = %s", (omgang_id,))
    row = cur.fetchone()
    return int(row[0]) if row else 0


def parse_constraints(args_allow: list[str], args_ban: list[str], length: int) -> list[set[str]]:
    allowed = [set(["1", "X", "2"]) for _ in range(length)]

    def apply_allow(expr: str):
        # format: pos=values  e.g., 1=2  or 5=1X  or 7=12
        if "=" not in expr:
            raise ValueError(f"Ogiltig --allow: '{expr}', förväntar pos=values")
        pos_s, vals_s = expr.split("=", 1)
        pos = int(pos_s)
        if not (1 <= pos <= length):
            raise ValueError(f"Position {pos} utanför [1,{length}]")
        vals = set(c.upper() for c in vals_s if c.upper() in ("1", "X", "2"))
        if not vals:
            raise ValueError(f"--allow {expr}: inga giltiga tecken (1/X/2)")
        allowed[pos - 1] = vals

    def apply_ban(expr: str):
        # format: pos=values  e.g., 5=X  or 9=12  or 3=1X
        if "=" not in expr:
            raise ValueError(f"Ogiltig --ban: '{expr}', förväntar pos=values")
        pos_s, vals_s = expr.split("=", 1)
        pos = int(pos_s)
        if not (1 <= pos <= length):
            raise ValueError(f"Position {pos} utanför [1,{length}]")
        vals = set(c.upper() for c in vals_s if c.upper() in ("1", "X", "2"))
        if not vals:
            return
        allowed[pos - 1] -= vals
        if not allowed[pos - 1]:
            raise ValueError(f"--ban {expr}: inga återstående tillåtna tecken i position {pos}")

    for a in args_allow or []:
        apply_allow(a)
    for b in args_ban or []:
        apply_ban(b)
    return allowed


def build_regex(allowed: list[set[str]]) -> str:
    parts: list[str] = []
    for s in allowed:
        if not s or s == {"1", "X", "2"}:
            parts.append("[12X]")
        else:
            chars = "".join(sorted(s, key=lambda c: {"1": 0, "X": 1, "2": 2}[c]))
            parts.append(f"[{chars}]")
    return "^" + "".join(parts) + "$"


def generate_kuponger(omgang_id: int, allow: list[str], ban: list[str], limit: int | None, clear: bool, dry_run: bool) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            length = get_match_count(cur, omgang_id)
            if length <= 0:
                raise RuntimeError(f"Hittade inga matcher för omgang_id={omgang_id}")
            allowed = parse_constraints(allow, ban, length)
            regex = build_regex(allowed)

            # Bygg WHERE‑villkor dynamiskt
            where_clauses = ["c.omgang_id = %s", "k.rad ~ %s"]
            where_params: list[object] = [omgang_id, regex]
            # Filter för oddset_right_count (inklusive gränser)
            if args_oddset_min is not None:
                where_clauses.append("c.oddset_right_count >= %s")
                where_params.append(args_oddset_min)
            if args_oddset_max is not None:
                where_clauses.append("c.oddset_right_count <= %s")
                where_params.append(args_oddset_max)

            if dry_run:
                where_sql = " AND ".join(where_clauses)
                count_sql = f"""
                    SELECT COUNT(*)
                    FROM kommande c
                    JOIN kombinationer k ON k.kombinations_id = c.rad
                    WHERE {where_sql}
                """
                cur.execute(count_sql, tuple(where_params))
                count = int(cur.fetchone()[0])
                print(f"Dry-run: {count} rader matchar filtret för omgång {omgang_id}")
                return

            if clear:
                cur.execute("DELETE FROM kuponger WHERE omgang_id = %s", (omgang_id,))
                conn.commit()
                print(f"Rensade befintliga kuponger för omgång {omgang_id}")

            where_sql = " AND ".join(where_clauses)
            select_sql = f"""
                SELECT %s, k.kombinations_id
                FROM kommande c
                JOIN kombinationer k ON k.kombinations_id = c.rad
                WHERE {where_sql}
                ORDER BY c.rank_predict_1 DESC NULLS LAST, c.id
            """
            # Första param i SELECT är konstant omgang_id som infogas i kuponger
            params = (omgang_id,) + tuple(where_params)
            if limit is not None and limit > 0:
                select_sql += " LIMIT %s"
                params = params + (limit,)

            insert_sql = f"INSERT INTO kuponger (omgang_id, rad) {select_sql} ON CONFLICT (omgang_id, rad) DO NOTHING"
            cur.execute(insert_sql, params)
            affected = cur.rowcount if cur.rowcount is not None else 0
            conn.commit()
            print(f"Kuponger: insatta/ignorerade rader (rowcount={affected}) för omgång {omgang_id}")


def main():
    ap = argparse.ArgumentParser(description="Generera kuponger för en omgång baserat på constraints")
    ap.add_argument("--omgang-id", "--omgang_id", dest="omgang_id", type=int, required=True, help="Omgang ID")
    ap.add_argument("--allow", action="append", help="Tillåtna tecken per position, ex '1=2' eller '5=1X'. Kan upprepas.")
    ap.add_argument("--ban", action="append", help="Förbjudna tecken per position, ex '5=X' eller '9=1'. Kan upprepas.")
    ap.add_argument("--limit", type=int, default=None, help="Max antal rader att infoga")
    ap.add_argument("--clear", action="store_true", help="Rensa befintliga kuponger för omgången före insert")
    ap.add_argument("--dry-run", action="store_true", help="Beräkna och skriv endast hur många rader som matchar")
    ap.add_argument("--oddset-right-count-min", dest="oddset_min", type=int, default=None, help="Min oddset_right_count (inklusive)")
    ap.add_argument("--oddset-right-count-max", dest="oddset_max", type=int, default=None, help="Max oddset_right_count (inklusive)")
    args = ap.parse_args()

    # Vidarebefordra filtren till generate_kuponger via closures/yt‑variabler
    global args_oddset_min, args_oddset_max
    args_oddset_min = args.oddset_min
    args_oddset_max = args.oddset_max

    generate_kuponger(args.omgang_id, args.allow, args.ban, args.limit, args.clear, args.dry_run)


if __name__ == "__main__":
    main()
