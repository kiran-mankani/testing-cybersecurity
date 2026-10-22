"""FastAPI service for the NSL-KDD intrusion-detection model.

Run from the project root:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pymongo.errors import PyMongoError

from backend import config, database
from backend.model_service import ModelNotLoaded, ModelService
from backend.schemas import BatchRequest, BatchResponse, ConnectionFeatures, PredictResponse

service = ModelService()


@asynccontextmanager
async def lifespan(app):
    service.load()
    yield


app = FastAPI(title="Network Intrusion Detection API", version="1.0.0", lifespan=lifespan)


def _require_model():
    if not service.ready:
        raise HTTPException(status_code=503, detail=service.error)


def _documents(records, predictions, source):
    now = datetime.now(timezone.utc)
    model = service.metadata["model_name"]
    return [
        {"timestamp": now, "source": source, "model": model, "features": rec, **pred}
        for rec, pred in zip(records, predictions)
    ]


def _run(records, store, source, explain=False):
    _require_model()
    try:
        predictions, elapsed_ms = service.predict(records, explain=explain)
    except ModelNotLoaded as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    stored = database.save_predictions(_documents(records, predictions, source)) if store else 0
    return predictions, elapsed_ms, stored


@app.get("/")
def root():
    return {"message": "Intrusion Detection API is running", "docs": "/docs"}


@app.get("/health")
def health():
    db_ok, db_detail = database.ping()
    return {
        "model_loaded": service.ready,
        "model_error": service.error,
        "model": service.metadata["model_name"] if service.ready else None,
        "database_connected": db_ok,
        "database": db_detail,
    }


@app.get("/model/info")
def model_info():
    _require_model()
    meta = service.metadata
    return {k: meta[k] for k in (
        "model_name", "anomaly_detector", "if_target_false_alarm_rate", "training_samples",
        "feature_count", "class_count", "feature_names", "categorical_features",
        "categories", "classes", "xgb_params", "training_method", "preprocessing", "trained_at",
    )}


@app.get("/model/performance")
def model_performance():
    """Evaluation results as saved by src/evaluation/ (not recomputed here)."""
    out = {}
    for name in ("cv_summary", "official_split_binary", "hybrid_official_split", "hybrid_cv"):
        path = config.FAIR_COMPARISON_DIR / f"{name}.csv"
        if path.exists():
            out[name] = {
                "source": str(path.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
                "rows": pd.read_csv(path).to_dict(orient="records"),
            }
    if not out:
        raise HTTPException(status_code=404, detail="No results found in results/fair_comparison/")
    return out


@app.post("/predict", response_model=PredictResponse)
def predict(features: ConnectionFeatures, store: bool = True, source: str = "form",
            explain: bool = True):
    predictions, elapsed_ms, stored = _run([features.model_dump()], store, source, explain)
    return {"prediction": predictions[0], "model": service.metadata["model_name"],
            "elapsed_ms": elapsed_ms, "stored": stored == 1}


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if not request.records:
        raise HTTPException(status_code=422, detail="records is empty")
    if len(request.records) > config.MAX_BATCH_ROWS:
        raise HTTPException(status_code=413, detail=f"At most {config.MAX_BATCH_ROWS} records per request")
    records = [r.model_dump() for r in request.records]
    predictions, elapsed_ms, stored = _run(records, request.store, request.source)
    return {"predictions": predictions, "model": service.metadata["model_name"],
            "elapsed_ms": elapsed_ms, "stored": stored}


@lru_cache(maxsize=1)
def _test_rows():
    return pd.read_csv(config.RAW_TEST_CSV)


@app.get("/examples")
def examples(label: str = None, n: int = Query(1, ge=1, le=100), seed: int = None):
    """Real records from data/raw/kdd_test.csv, with their true label, for demos."""
    if not config.RAW_TEST_CSV.exists():
        raise HTTPException(status_code=404, detail=f"{config.RAW_TEST_CSV} not found")
    rows = _test_rows()
    if label:
        rows = rows[rows["labels"] == label]
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"No test rows with label '{label}'")
    picked = rows.sample(n=min(n, len(rows)), random_state=seed)
    return {
        "labels_available": sorted(_test_rows()["labels"].unique().tolist()),
        "records": picked.drop(columns=["num_outbound_cmds"]).to_dict(orient="records"),
    }


@app.get("/predictions")
def predictions(limit: int = Query(20, ge=1, le=500)):
    try:
        return database.recent_predictions(limit)
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB unavailable: {exc}")


@app.get("/predictions/stats")
def predictions_stats():
    try:
        return database.prediction_stats()
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB unavailable: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=config.API_HOST, port=config.API_PORT)
