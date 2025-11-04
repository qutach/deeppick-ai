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


def migrate(cur):
    cur.execute(
        """
        ALTER TABLE kommande
        ADD COLUMN IF NOT EXISTS correct INTEGER
        """
    )


def main():
    ap = argparse.ArgumentParser(description="Add column 'correct' to kommande (idempotent)")
    args = ap.parse_args()
    with get_conn() as conn, conn.cursor() as cur:
        migrate(cur)
        conn.commit()
    print("Migration complete: kommande.correct added (if missing)")


if __name__ == "__main__":
    main()

