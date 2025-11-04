import os
import argparse
from collections import Counter
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


def fetch_outcome_and_len(cur, omgang_id: int) -> tuple[str, int]:
    cur.execute(
        """
        SELECT k.rad
        FROM omgang o
        JOIN kombinationer k ON k.kombinations_id = o.correct
        WHERE o.omgang_id = %s
        """,
        (omgang_id,),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError(
            f"Saknar korrekt rad (omgang.correct) för omgang_id={omgang_id}. Avbryter."
        )
    outcome = sanitize_rad(row[0])
    # Längd enligt antal matcher i denna omgång
    cur.execute("SELECT COUNT(*) FROM omgangsmatch WHERE omgang_id = %s", (omgang_id,))
    mrow = cur.fetchone()
    match_count = int(mrow[0]) if mrow else 0
    if match_count <= 0:
        raise RuntimeError(f"Hittade inga matcher för omgang_id={omgang_id}")
    # Trimma outcome till match_count (säkerhetsbälte)
    outcome = outcome[:match_count]
    return outcome, match_count


def iter_exported_rader_for_omgang(outdir: str, omgang_id: int):
    prefix = f"{omgang_id}-"
    for name in sorted(os.listdir(outdir)):
        if not name.lower().endswith(".txt"):
            continue
        if not name.startswith(prefix):
            continue
        path = os.path.join(outdir, name)
        yield path


def parse_export_line_to_rad(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        return None
    # Hoppa över första kolumnen om den är 'E'
    if parts[0].upper() == "E":
        parts = parts[1:]
    # Behåll endast 1/X/2 och bygg sträng
    tokens = [p.upper() for p in parts if p.upper() in ("1", "X", "2")]
    if not tokens:
        return None
    return "".join(tokens)


def ratta_exporterade(omgang_id: int, outdir: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        outcome, match_count = fetch_outcome_and_len(cur, omgang_id)

    files = list(iter_exported_rader_for_omgang(outdir, omgang_id))
    if not files:
        raise RuntimeError(f"Hittade inga exporterade txt‑filer i {outdir} för omgang_id={omgang_id}")

    dist = Counter()  # key = antal rätt (0..match_count)
    total_rows = 0
    processed_files = 0

    for path in files:
        processed_files += 1
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Första raden är gametype, hoppa över (kan följas av tomrad)
        data_lines = lines[1:]
        for line in data_lines:
            rad = parse_export_line_to_rad(line)
            if not rad:
                continue
            s = sanitize_rad(rad)[:match_count]
            # Räkna rätt över överlappande längd
            L = min(len(s), len(outcome))
            correct = sum(1 for i in range(L) if s[i] == outcome[i])
            dist[correct] += 1
            total_rows += 1

    print(f"Rättade {total_rows} rader från {processed_files} fil(er) för omgang {omgang_id}.")
    print("Fördelning (antal rader per rättnivå):")
    for r in range(match_count, -1, -1):
        cnt = dist.get(r, 0)
        pct = (100.0 * cnt / total_rows) if total_rows > 0 else 0.0
        print(f"{r} rätt: {cnt} st ({pct:.2f}%)")


def main():
    ap = argparse.ArgumentParser(description="Rätta exporterade kupong‑txt för en given omgång mot omgang.correct")
    ap.add_argument("--omgang-id", "--omgang_id", dest="omgang_id", type=int, required=True, help="Omgang ID")
    ap.add_argument("--outdir", type=str, default="/app/svenskaspel", help="Katalog med exporterade txt‑filer (default /app/svenskaspel)")
    args = ap.parse_args()

    ratta_exporterade(args.omgang_id, args.outdir)


if __name__ == "__main__":
    main()

