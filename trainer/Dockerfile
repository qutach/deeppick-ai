FROM python:3.13-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY train.py /app/train.py
COPY lightgbm_train.py /app/lightgbm_train.py
COPY optuna_lgbm.py /app/optuna_lgbm.py
COPY optuna_lgbm_gpu.py /app/optuna_lgbm_gpu.py
COPY optuna_lgbm_topk.py /app/optuna_lgbm_topk.py
COPY optuna_lgbm_topk_bias.py /app/optuna_lgbm_topk_bias.py

CMD ["tail", "-f", "/dev/null"]

