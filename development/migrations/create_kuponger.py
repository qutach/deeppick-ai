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
        CREATE TABLE IF NOT EXISTS kuponger (
            id SERIAL PRIMARY KEY,
            omgang_id INTEGER NOT NULL,
            rad INTEGER NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (omgang_id, rad),
            FOREIGN KEY (omgang_id) REFERENCES omgang(omgang_id) ON DELETE CASCADE,
            FOREIGN KEY (rad) REFERENCES kombinationer(kombinations_id) ON DELETE RESTRICT
        )
        """
    )


def main():
    ap = argparse.ArgumentParser(description="Create table 'kuponger' if missing (idempotent)")
    ap.parse_args()
    with get_conn() as conn, conn.cursor() as cur:
        migrate(cur)
        conn.commit()
    print("Migration complete: table 'kuponger' ensured")


if __name__ == "__main__":
    main()

