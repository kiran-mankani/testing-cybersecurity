"""Request/response models. Field names and order match the 40 model features
(NSL-KDD minus the constant num_outbound_cmds)."""
from typing import Annotated, List

from pydantic import BaseModel, ConfigDict, Field

Count = Annotated[float, Field(ge=0)]
Rate = Annotated[float, Field(ge=0, le=1)]
Binary = Annotated[int, Field(ge=0, le=1)]


class ConnectionFeatures(BaseModel):
    model_config = ConfigDict(extra="ignore")

    duration: Count
    protocol_type: str
    service: str
    flag: str
    src_bytes: Count
    dst_bytes: Count
    land: Binary
    wrong_fragment: Count
    urgent: Count
    hot: Count
    num_failed_logins: Count
    logged_in: Binary
    num_compromised: Count
    root_shell: Count
    su_attempted: Count
    num_root: Count
    num_file_creations: Count
    num_shells: Count
    num_access_files: Count
    is_host_login: Binary
    is_guest_login: Binary
    count: Count
    srv_count: Count
    serror_rate: Rate
    srv_serror_rate: Rate
    rerror_rate: Rate
    srv_rerror_rate: Rate
    same_srv_rate: Rate
    diff_srv_rate: Rate
    srv_diff_host_rate: Rate
    dst_host_count: Count
    dst_host_srv_count: Count
    dst_host_same_srv_rate: Rate
    dst_host_diff_srv_rate: Rate
    dst_host_same_src_port_rate: Rate
    dst_host_srv_diff_host_rate: Rate
    dst_host_serror_rate: Rate
    dst_host_srv_serror_rate: Rate
    dst_host_rerror_rate: Rate
    dst_host_srv_rerror_rate: Rate


class BatchRequest(BaseModel):
    records: List[ConnectionFeatures]
    store: bool = True
    source: str = "batch"


class ClassProbability(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(alias="class")
    probability: float


class Prediction(BaseModel):
    predicted_class: str
    confidence: float
    top_classes: List[ClassProbability]
    classifier_says_attack: bool
    anomaly_score: float
    anomaly_threshold: float
    anomaly_flag: bool
    alert: bool
    verdict: str
    severity: str
    response: str
    warnings: List[str]


class PredictResponse(BaseModel):
    prediction: Prediction
    model: str
    elapsed_ms: float
    stored: bool


class BatchResponse(BaseModel):
    predictions: List[Prediction]
    model: str
    elapsed_ms: float
    stored: int
