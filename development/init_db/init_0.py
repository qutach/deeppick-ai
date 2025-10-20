import os
import psycopg2


def get_conn():
    """Skapar och returnerar en psycopg2-anslutning med miljövariabler.

    Kräver: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
    Valfritt: POSTGRES_HOST (default 'localhost'), POSTGRES_PORT (default '5432')
    """
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ["POSTGRES_DB"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return psycopg2.connect(dbname=db, user=user, password=password, host=host, port=port)


def skapa_tabell_kombinationer(cur):
    """Skapar tabellen `kombinationer` om den inte finns."""
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS kombinationer (
            kombinations_id SERIAL PRIMARY KEY,
            rad TEXT NOT NULL UNIQUE
        )
        """
    )


def skapa_tabell_omgang(cur):
    """Skapar tabellen `omgang` med kolumnerna enligt specifikationen.

    Kolumner:
      - omgang_id: SERIAL PRIMARY KEY
      - year: INTEGER
      - week: INTEGER
      - datatype: TEXT
      - svspelinfo_id: INTEGER
      - gametype: TEXT
      - correct: INTEGER som FOREIGN KEY -> kombinationer(kombinations_id)
    """
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS omgang (
            omgang_id SERIAL PRIMARY KEY,
            year INTEGER,
            week INTEGER,
            datatype TEXT,
            svspelinfo_id INTEGER,
            gametype TEXT,
            correct INTEGER,
            -- Nya kolumner för radsummor (läggs även via ALTER nedan om tabellen redan finns)
            oddset_radsumma_max REAL,
            oddset_radsumma_min REAL,
            svenska_folket_radsumma_max REAL,
            svenska_folket_radsumma_min REAL,
            FOREIGN KEY (correct) REFERENCES kombinationer(kombinations_id)
        )
        """
    )
    # Säkerställ kolumner finns även i befintlig tabell
    cur.execute(
        """
        ALTER TABLE omgang
        ADD COLUMN IF NOT EXISTS oddset_radsumma_max REAL,
        ADD COLUMN IF NOT EXISTS oddset_radsumma_min REAL,
        ADD COLUMN IF NOT EXISTS svenska_folket_radsumma_max REAL,
        ADD COLUMN IF NOT EXISTS svenska_folket_radsumma_min REAL
        """
    )


def skapa_tabell_omgangsmatch(cur):
    """Skapar tabellen `omgangsmatch` med kolumner enligt användarens lista.

    Datatyper (antaganden):
      - matchnummer: INTEGER
      - hemmalag: TEXT
      - bortalag: TEXT
      - oddset_rank: INTEGER
      - people_rank: INTEGER
      - oddset1: REAL
      - oddsetx: REAL
      - oddset2: REAL
      - oddset_procent1: REAL
      - oddset_procentx: REAL
      - oddset_procent2: REAL
      - svenska_folket_odds1: REAL
      - svenska_folket_oddsx: REAL
      - svenska_folket_odds2: REAL
      - svenska_folket_procent1: REAL
      - svenska_folket_procentx: REAL
      - svenska_folket_procent2: REAL
      - tio_tidningar1: REAL
      - tio_tidningarx: REAL
      - tio_tidningar2: REAL
    """
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS omgangsmatch (
            omgangsmatch_id SERIAL PRIMARY KEY,
            omgang_id INTEGER NOT NULL,
            matchnummer INTEGER NOT NULL,
            hemmalag TEXT,
            bortalag TEXT,
            oddset_rank INTEGER,
            people_rank INTEGER,
            oddset1 REAL,
            oddsetx REAL,
            oddset2 REAL,
            oddset_procent1 REAL,
            oddset_procentx REAL,
            oddset_procent2 REAL,
            svenska_folket_odds1 REAL,
            svenska_folket_oddsx REAL,
            svenska_folket_odds2 REAL,
            svenska_folket_procent1 REAL,
            svenska_folket_procentx REAL,
            svenska_folket_procent2 REAL,
            tio_tidningar1 INTEGER CHECK (tio_tidningar1 >= 0 AND tio_tidningar1 <= 10),
            tio_tidningarx INTEGER CHECK (tio_tidningarx >= 0 AND tio_tidningarx <= 10),
            tio_tidningar2 INTEGER CHECK (tio_tidningar2 >= 0 AND tio_tidningar2 <= 10),
            tio_tidningar_present BOOLEAN DEFAULT FALSE,
            UNIQUE (omgang_id, matchnummer)
        )
        """
    )
    # Lägg till foreign key-constraint separat (säkerställer att tabellen finns innan vi länkar)
    cur.execute(
        """
        ALTER TABLE omgangsmatch
        ADD CONSTRAINT omgangsmatch_omgang_fk FOREIGN KEY (omgang_id)
            REFERENCES omgang(omgang_id) ON DELETE RESTRICT
        """
    )
    # Skapa ett index för snabbare JOINs på omgang_id
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_omgangsmatch_omgang_id ON omgangsmatch(omgang_id)
        """
    )

def skapa_tabell_kommande(cur):
    """Skapar tabellen `kommande` som lagrar rader (kombinationer) per omgång
    och deras sammanlagda radsummor för oddset och svenska folket."""
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS kommande (
            id SERIAL PRIMARY KEY,
            omgang_id INTEGER NOT NULL,
            rad INTEGER NOT NULL,
            oddset_radsumma REAL,
            svenska_folket_radsumma REAL,
            UNIQUE (omgang_id, rad),
            FOREIGN KEY (omgang_id) REFERENCES omgang(omgang_id) ON DELETE RESTRICT,
            FOREIGN KEY (rad) REFERENCES kombinationer(kombinations_id) ON DELETE RESTRICT
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_kommande_omgang_id ON kommande(omgang_id)
        """
    )


def initiera_databas():
    with get_conn() as conn:
        with conn.cursor() as cur:
            skapa_tabell_kombinationer(cur)
            skapa_tabell_omgang(cur)
            skapa_tabell_omgangsmatch(cur)
            skapa_tabell_kommande(cur)
        conn.commit()
    print("Tabellen 'kombinationer' är skapad eller fanns redan.")


if __name__ == "__main__":
    initiera_databas()
