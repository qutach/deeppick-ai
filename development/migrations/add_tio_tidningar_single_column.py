import os
import psycopg2


def get_conn():
    return psycopg2.connect(
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
    )


def add_columns(cur, table: str):
    cols_sql = ",\n        ".join(
        [f"ADD COLUMN IF NOT EXISTS rank{i}_oddset_tio_tidningar INTEGER" for i in range(1, 14)]
    )
    cur.execute(
        f"""
        ALTER TABLE {table}
        {cols_sql}
        """
    )


def backfill(cur, table: str):
    for i in range(1, 14):
        cur.execute(
            f"""
            UPDATE {table}
            SET rank{i}_oddset_tio_tidningar = COALESCE(
                rank{i}_oddset_tio_tidningar,
                rank{i}_oddset_tio_tidningar1,
                rank{i}_oddset_tio_tidningarx,
                rank{i}_oddset_tio_tidningar2
            )
            """
        )


def main():
    with get_conn() as conn, conn.cursor() as cur:
        for table in ("kommande", "historik"):
            add_columns(cur, table)
            backfill(cur, table)
        conn.commit()
    print("Added single per-rank tio_tidningar columns and backfilled in kommande and historik")


if __name__ == "__main__":
    main()

