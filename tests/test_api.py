"""API tests. MongoDB is replaced by an in-memory stub; the real model bundle is used.

Run from the project root:  python -m pytest tests -q
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import config, database, main

pytestmark = pytest.mark.skipif(
    not config.MODEL_BUNDLE_PATH.exists(),
    reason="run python -m src.models.train_deployment_model first",
)


@pytest.fixture()
def stored(monkeypatch):
    docs = []
    monkeypatch.setattr(database, "save_predictions", lambda d: docs.extend(d) or len(d))
    monkeypatch.setattr(database, "ping", lambda: (True, "stub"))
    return docs


@pytest.fixture()
def client(stored):
    with TestClient(main.app) as c:
        yield c


def rows_for(label, n=1):
    rows = pd.read_csv(config.RAW_TEST_CSV)
    rows = rows[rows["labels"] == label].head(n)
    return rows.drop(columns=["labels", "num_outbound_cmds"]).to_dict(orient="records")


def test_health(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model"] == "Standard Weighted XGBoost"


def test_predict_neptune_is_attack(client, stored):
    res = client.post("/predict", json=rows_for("neptune")[0])
    assert res.status_code == 200
    body = res.json()
    pred = body["prediction"]
    assert pred["predicted_class"] == "neptune"
    assert pred["alert"] is True and pred["verdict"] == "Attack"
    assert pred["severity"] == "High" and pred["response"] == "Investigate"
    assert 0 <= pred["confidence"] <= 1
    assert body["stored"] is True and len(stored) == 1
    assert stored[0]["predicted_class"] == "neptune"


def test_predict_without_storing(client, stored):
    res = client.post("/predict", params={"store": False}, json=rows_for("normal")[0])
    assert res.status_code == 200
    assert res.json()["stored"] is False and stored == []


def test_unseen_category_is_encoded_and_reported(client):
    record = rows_for("normal")[0] | {"service": "not_a_service"}
    pred = client.post("/predict", json=record).json()["prediction"]
    assert any("not_a_service" in w for w in pred["warnings"])


def test_invalid_rate_is_rejected(client):
    record = rows_for("normal")[0] | {"serror_rate": 1.5}
    assert client.post("/predict", json=record).status_code == 422


def test_batch_matches_single(client):
    records = rows_for("smurf", 3) + rows_for("normal", 3)
    batch = client.post("/predict/batch", json={"records": records, "store": False}).json()
    assert len(batch["predictions"]) == 6
    single = client.post("/predict", params={"store": False}, json=records[0]).json()
    assert batch["predictions"][0] == single["prediction"]


def test_examples_have_all_features(client):
    body = client.get("/examples", params={"label": "satan", "n": 2}).json()
    assert len(body["records"]) == 2
    assert set(main.service.metadata["feature_names"]) <= set(body["records"][0])
