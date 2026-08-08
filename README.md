[README.md](https://github.com/user-attachments/files/30859311/README.md)
# Mini Production ML System — Customer Churn Prediction

Binary classification: predict telecom customer churn. Built for the "Design and Build a
Mini Production ML System" assignment. See `docs/design_doc.md` for the full writeup and
`docs/architecture_diagram.png` for the system diagram.

## Repo layout

```
data/
  raw/                 downloaded source CSV + simulated daily drops
  processed/           training_table.csv (built by ingestion), ingestion_log.jsonl
features/
  feature_engineering.py   SHARED module used by both the training notebook and serving API
ingestion/
  ingest.ipynb          micro-batch ingestion notebook
training/
  train.ipynb            load -> split -> train baseline+candidate -> evaluate -> save
serving/
  app.py                 FastAPI service (/predict, /health) — a live service, kept as .py
  schemas.py              request/response models
  load_test.ipynb          latency/throughput measurement notebook (requires the API running)
monitoring/
  drift_check.ipynb        data quality + drift check notebook
  retrain_trigger.ipynb     retraining decision logic notebook
tests/
  test_features.py, test_api.py    pytest suite, kept as .py to run under CI
configs/
  config.yaml
models/                 saved model artifacts (latest.joblib + timestamped versions)
artifacts/eval/         evaluation reports, latency report, sample drift report
docs/
  design_doc.md, architecture_diagram.png
Dockerfile
requirements.txt
```

**Why some pieces are notebooks and some aren't:** `ingest.ipynb`, `train.ipynb`,
`drift_check.ipynb`, and `retrain_trigger.ipynb` are exploratory/pipeline-run artifacts —
notebooks make sense there, and the assignment explicitly allows the training pipeline to
be a notebook. `feature_engineering.py` stays a plain module because it's imported
identically by both the training notebook and the live FastAPI service — that shared
import is the project's guard against training-serving skew, and a notebook can't be
cleanly imported that way. `serving/app.py` stays a script because it has to run as a
persistent process under `uvicorn`, not execute top-to-bottom once. `tests/` stay `.py`
so they run under plain `pytest`.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Ingest data — open ingestion/ingest.ipynb, set INPUT_FILE, run all cells.
#    Run once for the initial batch, then again per new daily file.
jupyter nbconvert --to notebook --execute --inplace ingestion/ingest.ipynb

# 2. Train — open training/train.ipynb and run all cells (or via nbconvert):
jupyter nbconvert --to notebook --execute --inplace training/train.ipynb

# 3. Serve
uvicorn serving.app:app --port 8000
# in another terminal:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json

# 4. Measure latency — with the API running in another terminal:
jupyter nbconvert --to notebook --execute --inplace serving/load_test.ipynb

# 5. Check for drift on a new batch — edit RECENT_PATH in the notebook, then:
jupyter nbconvert --to notebook --execute --inplace monitoring/drift_check.ipynb

# 6. Run tests
pytest tests/ -v
```

All five notebooks can also just be opened in Jupyter/JupyterLab and run cell-by-cell —
`jupyter notebook` from the project root, or open them in VS Code / Google Colab.

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-api
```
