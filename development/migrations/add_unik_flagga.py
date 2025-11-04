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


def ensure_columns(cur):
    cur.execute(
        """
        ALTER TABLE historik
        ADD COLUMN IF NOT EXISTS unik_flagga BOOLEAN
        """
    )
    cur.execute(
        """
        ALTER TABLE kommande
        ADD COLUMN IF NOT EXISTS unik_flagga BOOLEAN
        """
    )


def update_historik(cur):
    # Directed uniqueness: TRUE if no older (higher omgang_id) historik with same rad and correct >= 12; else FALSE
    cur.execute(
        """
        UPDATE historik h
        SET unik_flagga = CASE
                            WHEN EXISTS (
                                SELECT 1 FROM historik hh
                                WHERE hh.rad = h.rad
                                  AND hh.correct >= 12
                                  AND hh.omgang_id > h.omgang_id
                            ) THEN FALSE
                            ELSE TRUE
                          END
        """
    )


def update_kommande(cur, limit_to_omgang: int | None):
    params = []
    where = ""
    if limit_to_omgang is not None:
        where = "WHERE c.omgang_id = %s"
        params.append(limit_to_omgang)
    # Directed uniqueness relative to each kommande row's omgång
    cur.execute(
        f"""
        UPDATE kommande c
        SET unik_flagga = CASE
                            WHEN EXISTS (
                                SELECT 1 FROM historik h
                                WHERE h.rad = c.rad
                                  AND h.correct >= 12
                                  AND h.omgang_id > c.omgang_id
                            ) THEN FALSE
                            ELSE TRUE
                          END
        {where}
        """,
        tuple(params),
    )


def main():
    ap = argparse.ArgumentParser(description="Add and backfill unik_flagga on historik and kommande")
    ap.add_argument("--omgang-id", type=int, default=None, help="Optional: only update kommande for this omgång")
    args = ap.parse_args()

    with get_conn() as conn, conn.cursor() as cur:
        ensure_columns(cur)
        update_historik(cur)
        update_kommande(cur, args.omgang_id)
        conn.commit()
    print("unik_flagga added/backfilled on historik and kommande")


if __name__ == "__main__":
    main()
