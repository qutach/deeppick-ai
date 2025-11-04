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


def fetch_outcome_rad(cur, omgang_id: int) -> str:
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
    if not row or row[0] is None or str(row[0]).strip() == "":
        raise RuntimeError(
            f"Saknar korrekt rad (omgang.correct) för omgang_id={omgang_id}. Avbryter."
        )
    # Säkerställ versaler för jämförelse
    return str(row[0]).upper()


def count_total_rows(cur, omgang_id: int) -> int:
    cur.execute("SELECT COUNT(*) FROM kommande WHERE omgang_id = %s", (omgang_id,))
    return int(cur.fetchone()[0])


def update_batch(conn, omgang_id: int, outcome_rad: str, id_batch: list[int]) -> int:
    if not id_batch:
        return 0
    with conn.cursor() as cur:
        # Set-baserad uppdatering för en given id‑batch, jämför teckenvis med outcome_rad
        # Jämför över positioner 1..len(outcome_rad); skydda för kortare kandidat med i <= length(k2.rad)
        sql = (
            """
            WITH calc AS (
                SELECT c2.id,
                       SUM(CASE WHEN gs.i <= length(k2.rad)
                                 AND substr(UPPER(%s), gs.i, 1) = substr(UPPER(k2.rad), gs.i, 1)
                                THEN 1 ELSE 0 END) AS corr
                FROM kommande c2
                JOIN kombinationer k2 ON k2.kombinations_id = c2.rad
                JOIN generate_series(1, length(%s)) AS gs(i) ON TRUE
                WHERE c2.id = ANY(%s)
                GROUP BY c2.id
            )
            UPDATE kommande c
            SET correct = calc.corr
            FROM calc
            WHERE c.id = calc.id
            """
        )
        cur.execute(sql, (outcome_rad, outcome_rad, id_batch))
    conn.commit()
    return len(id_batch)


def ratta_kommande(omgang_id: int, batch_size: int = 100_000) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            outcome_rad = fetch_outcome_rad(cur, omgang_id)
            total = count_total_rows(cur, omgang_id)
            print(f"Omgång {omgang_id}: outcome='{outcome_rad}', kommande‑rader={total}")

        processed = 0
        last_id = 0
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM kommande
                    WHERE omgang_id = %s AND id > %s
                    ORDER BY id
                    LIMIT %s
                    """,
                    (omgang_id, last_id, batch_size),
                )
                ids = [row[0] for row in cur.fetchall()]
            if not ids:
                break
            update_batch(conn, omgang_id, outcome_rad, ids)
            processed += len(ids)
            last_id = ids[-1]
            print(f"Rättar batch: {processed}/{total} uppdaterade")

        print(f"Klart: rättade {processed} rader i kommande för omgång {omgang_id}.")


def main():
    ap = argparse.ArgumentParser(description="Rätta kommande.correct mot omgångens outcome (kombinationer.rad)")
    ap.add_argument("--omgang-id", "--omgang_id", dest="omgang_id", type=int, required=True, help="Omgang ID")
    ap.add_argument("--batch-size", type=int, default=100_000, help="Batch-storlek (default 100000)")
    args = ap.parse_args()

    try:
        ratta_kommande(args.omgang_id, args.batch_size)
    except Exception as e:
        print(f"Fel: {e}")
        raise


if __name__ == "__main__":
    main()

