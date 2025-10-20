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


def print_rad_for_omgang(omgang_id: int):
    sql_rad = (
        """
        SELECT 
            k.rad,
            o.oddset_radsumma_max,
            o.oddset_radsumma_min,
            o.svenska_folket_radsumma_max,
            o.svenska_folket_radsumma_min
        FROM omgang o
        LEFT JOIN kombinationer k ON k.kombinations_id = o.correct
        WHERE o.omgang_id = %s
        """
    )
    sql_matches = (
        """
        SELECT
            matchnummer,
            hemmalag,
            bortalag,
            oddset_procent1,
            oddset_procentx,
            oddset_procent2,
            svenska_folket_procent1,
            svenska_folket_procentx,
            svenska_folket_procent2
        FROM omgangsmatch
        WHERE omgang_id = %s
        ORDER BY matchnummer ASC
        """
    )
    with get_conn() as conn, conn.cursor() as cur:
        # Hämta rad via correct
        cur.execute(sql_rad, (omgang_id,))
        row = cur.fetchone()
        if not row:
            print(f"Hittade ingen omgang med id={omgang_id}")
            return
        rad = row[0]
        stored_oddset_max = row[1]
        stored_oddset_min = row[2]
        stored_svenska_max = row[3]
        stored_svenska_min = row[4]

        # Hämta matcher
        cur.execute(sql_matches, (omgang_id,))
        matches = cur.fetchall()  # list of (matchnummer, hemmalag, bortalag, o1, ox, o2, s1, sx, s2)

        # Förbered och skriv ut rad överst (om finns)
        if rad is None:
            print("correct är NULL eller saknar match i kombinationer")
        else:
            # Filtrera bort allt utom 1/X/2 och normalisera X till versal
            filtered = ''.join(c.upper() for c in str(rad) if c.upper() in ("1", "X", "2"))
            match_count = len(matches)
            # Om fler tecken än antal matcher: anta att rad innehåller flera rader ihopslagna och skär av
            if match_count > 0 and len(filtered) != match_count:
                print(f"Observera: rad-längd {len(filtered)} matchar inte antal matcher {match_count}. Visar första {match_count}.")
            rad = filtered[:match_count] if match_count > 0 else filtered
            print(rad)

        # Skriv ut 13 matcher med utfall från rad och summera procentsatser
        sum_oddset = 0.0
        sum_svenska = 0.0
        for (
            matchnummer,
            hemmalag,
            bortalag,
            oddset_procent1,
            oddset_procentx,
            oddset_procent2,
            svenska_folket_procent1,
            svenska_folket_procentx,
            svenska_folket_procent2,
        ) in matches:
            pos = (int(matchnummer) - 1) if matchnummer is not None else None
            if isinstance(rad, str) and pos is not None and 0 <= pos < len(rad):
                utfall = rad[pos]
            else:
                utfall = "?"
            print(f"{matchnummer}. {hemmalag} - {bortalag}: {utfall}")

            # Summera radsumma baserat på utfall
            if utfall == "1":
                if oddset_procent1 is not None:
                    sum_oddset += float(oddset_procent1)
                if svenska_folket_procent1 is not None:
                    sum_svenska += float(svenska_folket_procent1)
            elif utfall == "X":
                if oddset_procentx is not None:
                    sum_oddset += float(oddset_procentx)
                if svenska_folket_procentx is not None:
                    sum_svenska += float(svenska_folket_procentx)
            elif utfall == "2":
                if oddset_procent2 is not None:
                    sum_oddset += float(oddset_procent2)
                if svenska_folket_procent2 is not None:
                    sum_svenska += float(svenska_folket_procent2)

        # Skriv ut radsummor
        print(f"Radsumma (oddset): {sum_oddset:.2f}")
        print(f"Radsumma (svenska_folket): {sum_svenska:.2f}")

        # Skriv ut referensvärden lagrade på omgang
        def fmt(x):
            return f"{float(x):.2f}" if x is not None else "NULL"

        print("— Referens (lagrat i omgang) —")
        print(f"oddset_radsumma_max: {fmt(stored_oddset_max)}")
        print(f"oddset_radsumma_min: {fmt(stored_oddset_min)}")
        print(f"svenska_folket_radsumma_max: {fmt(stored_svenska_max)}")
        print(f"svenska_folket_radsumma_min: {fmt(stored_svenska_min)}")


def main():
    parser = argparse.ArgumentParser(description="Skriv ut rad för given omgang via correct -> kombinationer")
    parser.add_argument("--omgang-id", type=int, default=1, help="Omgang ID (default 1)")
    args = parser.parse_args()
    print_rad_for_omgang(args.omgang_id)


if __name__ == "__main__":
    main()
