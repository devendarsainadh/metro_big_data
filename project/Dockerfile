FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir pandas numpy scikit-learn pyyaml pyspark jupyterlab ucimlrepo

CMD ["python", "scripts/run_pipeline.py", "--stage", "compare_etl_performance"]
