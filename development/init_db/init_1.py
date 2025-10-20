import os
import itertools
import tempfile
import psycopg2

def generate_rows():
    # Yield 3^13 rows in TEXT format (one value per line)
    signs = ['1', 'X', '2']
    for k in itertools.product(signs, repeat=13):
        yield ''.join(k) + '\n'

def get_conn():
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ["POSTGRES_DB"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return psycopg2.connect(dbname=db, user=user, password=password, host=host, port=port)

def upsert_combinations_with_copy():
    # Write to a temporary file to stream via COPY (fast)
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        for line in generate_rows():
            tmp.write(line)
        tmp.flush()
        tmp_name = tmp.name

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Ensure target table exists (rad must be UNIQUE)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kombinationer (
                    kombinations_id BIGSERIAL PRIMARY KEY,
                    rad TEXT NOT NULL UNIQUE
                );
            """)

            # Session knobs safe for bulk load
            cur.execute("SET synchronous_commit = OFF;")

            # Create a temp staging table (drops automatically at commit)
            cur.execute("CREATE TEMP TABLE staging_kombinationer (rad TEXT PRIMARY KEY) ON COMMIT DROP;")

            # COPY into staging (TEXT format = one value per line)
            copy_sql = "COPY staging_kombinationer (rad) FROM STDIN WITH (FORMAT text)"
            with open(tmp_name, 'r') as f:
                cur.copy_expert(copy_sql, f)

            # Insert only missing rows into kombinationer
            cur.execute("""
                INSERT INTO kombinationer (rad)
                SELECT s.rad
                FROM staging_kombinationer s
                ON CONFLICT (rad) DO NOTHING;
            """)

            # Update stats
            cur.execute("ANALYZE kombinationer;")

        conn.commit()

    print("Upsert done: ensured all combinations exist without truncating.")

if __name__ == "__main__":
    print("Ensuring 1,594,323 combinations exist (no truncate, FK-safe)...")
    upsert_combinations_with_copy()
