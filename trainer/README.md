# Trainer Usage

Den här mappen innehåller träningsskript för rankingmodeller (LightGBM och XGBoost). Kör dem normalt inuti `trainer`‑containern.

## Gemensamt

- Modellfiler sparas i `/models` (volymen mountas från `./models`).
- Featurelista sparas i `/models/feature_columns.txt` och måste matcha i prediktern.
- Använd `--save-model` (där det finns) för att spara tränad modell till `/models/lgbm_ranker.txt`.
- För XGBoost sparas modellen till `/models/xgb_ranker.json`.

## Skript

### optuna_lgbm.py

Hyperparameter‑tuning för LGBMRanker (allmän, utan top‑K‑fokus).

Flaggor (urval):
- `--trials`: antal Optuna‑trials.
- `--omgang_id_min`, `--omgang_id_max`: filtrering av historik.
- `--valid-frac`: andel omgångar i validering.
- `--early-stopping`: early stopping‑rundor.
- `--eval-at`: nivåer för NDCG (sista används för Optuna‑score).
- `--seed`: slumpfrö.
- `--save-model`: träna om på all data med bästa parametrar och spara modell.

Exempel:
```
python optuna_lgbm.py --trials 100 --eval-at 13 --save-model
```

### optuna_lgbm_topk.py

Top‑K‑fokuserad tuning (t.ex. topp‑500 per omgång) med NDCG@[K] och lambdarank‑truncation.

Extra flaggor:
- `--label-gain-scheme {default,balanced,conservative}`: viktning av labels i NDCG (default: balanced).
- `--truncation-levels`: kandidater för `lambdarank_truncation_level` (t.ex. `300 400 500 600`).
- `--topk`: K‑värde för sidomått (recall@K i logg). Påverkar inte Optuna‑score.

Exempel:
```
python optuna_lgbm_topk.py --trials 150 --eval-at 200 500 \
  --early-stopping 75 --label-gain-scheme balanced --save-model
```

### optuna_lgbm_topk_bias.py

Som `optuna_lgbm_topk.py` men med bias‑kontroll för `oddset_right_count` via transform + sample weights.

Extra flaggor:
- `--orc-transform {none,cap10,sqrt,log1p}`: transformera `oddset_right_count` (default `cap10`).
- `--weight-scheme {none,mild,medium}`: mjuk nedviktning för höga värden + boost för moderata värden med höga labels (default `mild`).

Övriga flaggor som i `optuna_lgbm_topk.py`.

Exempel:
```
python optuna_lgbm_topk_bias.py --trials 120 --eval-at 200 500 \
  --early-stopping 75 --label-gain-scheme balanced \
  --orc-transform cap10 --weight-scheme mild --save-model
```

Output:
- Modell: `/models/lgbm_ranker.txt`
- Param‑snapshot: `/models/lgbm_ranker_optuna_topk_bias.json` (inkl. transform/weights)

### lightgbm_train.py

Enkel LGBMRanker‑träning med fasta standardparametrar (ingen Optuna). Sparar modell och featurelista.

Exempel:
```
python lightgbm_train.py --omgang_id_min 1 --omgang_id_max 9999
```

### train.py (XGBoost)

Tränar XGBRanker (rank:ndcg) och sparar till `/models/xgb_ranker.json`.

Exempel:
```
python train.py --omgang_id_min 1 --omgang_id_max 9999
```

## Praktiska råd

- För topp‑500‑mål: använd `optuna_lgbm_topk.py` eller `optuna_lgbm_topk_bias.py` med `--eval-at 200 500` och truncation runt 400–500.
- Kontrollera recall@K i logg: `recall13@500` och `recall>=12@500` för att säkerställa att målen uppnås.
- Använd flera `--seed` vid tuning för robusthet. Träna sedan slutlig modell på all data med `--save-model`.

