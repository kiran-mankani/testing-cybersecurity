"""MongoDB storage for predictions.

The connection is lazy and failures are reported, not raised: the API keeps
predicting when MongoDB is down, it just cannot store or list history.
"""
from pymongo import DESCENDING, MongoClient
from pymongo.errors import PyMongoError

from backend import config

COLLECTION = "predictions"

_client = None


def _collection():
    global _client
    if _client is None:
        _client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=config.MONGODB_TIMEOUT_MS)
    return _client[config.MONGODB_DATABASE][COLLECTION]


def ping():
    """Return (ok, detail)."""
    try:
        _collection().database.client.admin.command("ping")
        return True, f"{config.MONGODB_URI}/{config.MONGODB_DATABASE}"
    except PyMongoError as exc:
        return False, str(exc).split(",")[0]


def save_predictions(documents):
    """Insert documents; return how many were stored (0 if MongoDB is unavailable)."""
    if not documents:
        return 0
    try:
        return len(_collection().insert_many(documents).inserted_ids)
    except PyMongoError:
        return 0


def recent_predictions(limit=20):
    cursor = _collection().find({}, {"features": 0}).sort("timestamp", DESCENDING).limit(limit)
    return [{**doc, "_id": str(doc["_id"])} for doc in cursor]


def prediction_stats():
    pipeline = [{"$group": {
        "_id": "$predicted_class",
        "count": {"$sum": 1},
        "alerts": {"$sum": {"$cond": ["$alert", 1, 0]}},
    }}, {"$sort": {"count": -1}}]
    by_class = [
        {"predicted_class": row["_id"], "count": row["count"], "alerts": row["alerts"]}
        for row in _collection().aggregate(pipeline)
    ]
    return {
        "total": sum(r["count"] for r in by_class),
        "alerts": sum(r["alerts"] for r in by_class),
        "by_class": by_class,
    }
