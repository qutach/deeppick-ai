import os
import argparse
from datetime import datetime
import psycopg2


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


def fetch_meta(cur, omgang_id: int) -> tuple[str, int]:
    cur.execute("SELECT gametype FROM omgang WHERE omgang_id = %s", (omgang_id,))
    row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError(f"Hittade inte gametype för omgang_id={omgang_id}")
    gametype = str(row[0])
    cur.execute("SELECT COUNT(*) FROM omgangsmatch WHERE omgang_id = %s", (omgang_id,))
    mrow = cur.fetchone()
    match_count = int(mrow[0]) if mrow else 0
    if match_count <= 0:
        raise RuntimeError(f"Hittade inga matcher för omgang_id={omgang_id}")
    return gametype, match_count


def fetch_rows(cur, omgang_id: int, limit: int | None):
    if limit is None:
        cur.execute(
            """
            SELECT k.rad
            FROM kuponger kp
            JOIN kombinationer k ON k.kombinations_id = kp.rad
            WHERE kp.omgang_id = %s
            ORDER BY kp.id
            """,
            (omgang_id,),
        )
    else:
        cur.execute(
            """
            SELECT k.rad
            FROM kuponger kp
            JOIN kombinationer k ON k.kombinations_id = kp.rad
            WHERE kp.omgang_id = %s
            ORDER BY random()
            LIMIT %s
            """,
            (omgang_id, limit),
        )
    return [r[0] for r in cur.fetchall()]


def export_kuponger(omgang_id: int, limit: int | None, outdir: str) -> str:
    with get_conn() as conn, conn.cursor() as cur:
        gametype, match_count = fetch_meta(cur, omgang_id)
        rows = fetch_rows(cur, omgang_id, limit)
        if not rows:
            raise RuntimeError(f"Inga rader i kuponger för omgang_id={omgang_id}")

    # Förbered statistik per position
    n = len(rows)
    counts = [{"1": 0, "X": 0, "2": 0} for _ in range(match_count)]

    # Skriv fil
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{omgang_id}-{ts}.txt"
    path = os.path.join(outdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{gametype}\n\n")
        for rad in rows:
            s = sanitize_rad(rad)
            s = s[:match_count]
            # uppdatera statistik
            for i, ch in enumerate(s):
                if ch in ("1", "X", "2"):
                    counts[i][ch] += 1
            # skriv raden
            parts = ["E"] + list(s)
            f.write(",".join(parts) + "\n")

    # Skriv statistik till stdout
    print(f"Skrev {n} rader till {path}")
    print("Statistik per position (antal och procent):")
    for i in range(match_count):
        c1 = counts[i]["1"]
        cx = counts[i]["X"]
        c2 = counts[i]["2"]
        def pct(x):
            return (100.0 * x / n) if n > 0 else 0.0
        print(
            f"Match {i+1}: 1={c1} ({pct(c1):.1f}%), X={cx} ({pct(cx):.1f}%), 2={c2} ({pct(c2):.1f}%)"
        )
    return path


def main():
    ap = argparse.ArgumentParser(description="Exportera kuponger till txt-fil i svenskaspel-katalogen")
    ap.add_argument("--omgang-id", "--omgang_id", dest="omgang_id", type=int, required=True, help="Omgang ID")
    ap.add_argument("--limit", type=int, default=None, help="Max antal rader att exportera (slumpat). Uteslut för alla.")
    ap.add_argument(
        "--outdir",
        type=str,
        default="/app/svenskaspel",
        help="Katalog för output-filen (default /app/svenskaspel)",
    )
    args = ap.parse_args()

    export_kuponger(args.omgang_id, args.limit, args.outdir)


if __name__ == "__main__":
    main()

