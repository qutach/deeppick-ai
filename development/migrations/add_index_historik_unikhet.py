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


def main():
    idx_name = "idx_historik_rad_omgang_ge12"
    sql = f"""
        CREATE INDEX IF NOT EXISTS {idx_name}
        ON historik(rad, omgang_id)
        WHERE correct >= 12
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
    print("Index created (or already existed):", idx_name)


if __name__ == "__main__":
    main()

