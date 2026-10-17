from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
from database import predictions_collection
import uvicorn

app = FastAPI(title="Network Intrusion Detection API")

# Input data ka structure (41 features ke liye aap isko expand kar sakti hain)
class NetworkFeatures(BaseModel):
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: int
    dst_bytes: int
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    # Baaki 29 features yahan add karein...

@app.get("/")
def read_root():
    return {"message": "Intrusion Detection API is running"}

@app.post("/predict")
async def predict(data: NetworkFeatures):
    try:
        # Yahan aap apna XGBoost model load karke predict karenge
        # Filhal dummy prediction:
        prediction = "Normal Traffic"
        confidence = 94.2
        class_probs = {
            "normal": 0.942, "neptune": 0.021, "satan": 0.012, 
            "ipsweep": 0.008, "portsweep": 0.006, "other": 0.011
        }
        
        # Database mein save karna
        record = {
            "timestamp": datetime.now(),
            "input_features": data.dict(),
            "prediction": prediction,
            "confidence": confidence
        }
        predictions_collection.insert_one(record)
        
        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "prediction_time": "0.023s"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent-predictions")
def get_recent_predictions():
    # Database se last 5 predictions fetch karna
    cursor = predictions_collection.find().sort("timestamp", -1).limit(5)
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)