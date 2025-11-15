# Predicter Usage

Den här mappen innehåller skript som läser scorer från tränade modeller och uppdaterar `kommande.rank_predict_1` i databasen. Kör dem normalt inuti `predicter`‑containern.

## Gemensamt

- Kräver modeller och featurelista i `/models`:
  - LightGBM: `/models/lgbm_ranker.txt`
  - XGBoost: `/models/xgb_ranker.json`
  - Features: `/models/feature_columns.txt`
- Skripten skapar kolumnen `rank_predict_1` i `kommande` om den saknas.
- Kör per batch; du kan begränsa till en viss omgång med flagga.

## Skript

### lightgbm_ranker.py

Läser LightGBM‑modellen och uppdaterar scorer.

Flaggor:
- `--omgang-id`: om satt, predicera endast för den omgången.
- `--batch-size`: antal rader per batch (default 5000).

Exempel:
```
python lightgbm_ranker.py --batch-size 5000
python lightgbm_ranker.py --omgang-id 4876
```

### lightgbm_ranker_bias.py

Som `lightgbm_ranker.py`, men applicerar samma `oddset_right_count`‑transform som i vissa tränare för train/predict‑paritet.

Extra flagga:
- `--orc-transform {none,cap10,sqrt,log1p}`: måste matcha transformen som användes vid träning (default `cap10`).

Exempel:
```
python lightgbm_ranker_bias.py --orc-transform cap10 --batch-size 5000
python lightgbm_ranker_bias.py --omgang-id 4876 --orc-transform sqrt
```

### ranker.py (XGBoost)

Predicter för XGBRanker.

Flaggor:
- `--omgang-id`: begränsa till en omgång.
- `--batch-size`: batchstorlek vid uppdatering.

Exempel:
```
python ranker.py --batch-size 5000
```

## Tips & drift

- Se till att predicterns transform (om du använder bias‑varianten) matchar tränarens, annars avviker poängen.
- För prestanda kan du öka `--batch-size` om minnet tillåter.
- Urval av topp‑K görs lämpligen per omgång med SQL/fönsterfunktioner efter att scorer skrivits.

