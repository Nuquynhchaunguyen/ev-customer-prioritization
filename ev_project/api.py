"""Local batch ranking API using the same trusted model as the Streamlit app."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .monitoring import evaluate_batch, load_reference
from .schema import FEATURE_COLUMNS, KNOWN_CATEGORIES, NUMERIC_FEATURES, ProfileValidationError, SCHEMA_VERSION
from .scoring import load_bundle, score_profiles

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model_bundle.joblib"
REFERENCE_PATH = ROOT / "models" / "monitoring_reference.json"
MAX_BATCH_SIZE = 5000


class CustomerProfile(BaseModel):
    """One synthetic profile. IDs remain text, including leading zeros."""
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    id: StrictStr = Field(min_length=1, max_length=128)
    Age: StrictInt = Field(ge=0, le=120)
    Annual_Income_USD: StrictInt = Field(ge=0, lt=2**63)
    Daily_Commute_km: float = Field(strict=True, ge=0, allow_inf_nan=False)
    Number_of_Cars_Owned: StrictInt = Field(ge=0, lt=2**63)
    Charging_Stations_Near_Home: StrictInt = Field(ge=0, lt=2**63)
    Charging_Stations_Near_Work: StrictInt = Field(ge=0, lt=2**63)
    Environmental_Concern_Level: StrictInt | StrictStr
    Gender: Literal["Female", "Male", "Other"]
    City_Type: Literal["Rural", "Suburban", "Urban"]
    Current_Car_Type: Literal["Hatchback", "SUV", "Sedan", "Truck"]
    Home_Charging_Possible: Literal["No", "Yes"]
    Subsidy_Available: Literal["No", "Yes"]
    Range_Anxiety_Level: Literal["High", "Low", "Medium"]


class RankingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profiles: list[CustomerProfile] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    k: StrictInt = Field(ge=1, le=MAX_BATCH_SIZE)


class RankedProfile(BaseModel):
    rank: int
    id: str
    yes_probability: float


class RankingResponse(BaseModel):
    model: str
    model_sha256: str
    schema_version: str
    count: int
    k: int
    ranking: list[RankedProfile]
    shortlist: list[RankedProfile]
    warnings: list[str]
    monitoring: dict
    score_meaning: str


def create_app(*, model_path: Path = MODEL_PATH, reference_path: Path = REFERENCE_PATH) -> FastAPI:
    """Paths are application configuration, never request parameters."""
    @asynccontextmanager
    async def lifespan(application):
        application.state.bundle = load_bundle(model_path)
        application.state.reference = load_reference(reference_path, model_path=model_path)
        application.state.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        application.state.reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        yield

    application = FastAPI(
        title="EV Customer Prioritization API",
        version="1.0.0",
        description=("Rank synthetic EV-interest profiles using the saved Logistic Regression pipeline. "
                     "Scores estimate the supplied Yes label and do not establish real purchase likelihood. "
                     "Local teaching prototype, without customer contact or persistent customer storage."),
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exception: RequestValidationError):
        # Do not echo submitted customer values in validation responses.
        errors = [{"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
                  for error in exception.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    def ensure_current_artifacts():
        try:
            current_model = hashlib.sha256(model_path.read_bytes()).hexdigest()
            current_reference = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        except OSError as error:
            raise HTTPException(status_code=503, detail="Model or monitoring reference is unavailable. Restore the matched files and restart the API.") from error
        if current_model != application.state.model_sha256 or current_reference != application.state.reference_sha256:
            raise HTTPException(status_code=503, detail="Model or monitoring reference changed after startup. Stop the API, rebuild a matching reference if retrained, and restart.")

    @application.get("/health", tags=["Readiness"])
    def health():
        ensure_current_artifacts()
        return {"status": "ok", "model": application.state.bundle["metadata"]["selected_model"],
                "model_sha256": application.state.model_sha256, "schema_version": SCHEMA_VERSION,
                "monitoring_reference_loaded": True}

    @application.get("/schema", tags=["Input contract"])
    def schema():
        return {"required_columns": ["id"] + FEATURE_COLUMNS, "numeric_features": NUMERIC_FEATURES,
                "known_categories": KNOWN_CATEGORIES, "maximum_batch_size": MAX_BATCH_SIZE,
                "id_rule": "Nonblank unique text; leading zeros preserved.",
                "k_rule": "Whole number from 1 to the number of supplied profiles.",
                "extra_columns": "Ignored before scoring; reported in warnings."}

    @application.post("/rank", response_model=RankingResponse, tags=["Ranking"],
                      responses={422: {"description": "Invalid profile, duplicate ID, JSON, or capacity"}})
    def rank(payload: RankingRequest):
        ensure_current_artifacts()
        frame = pd.DataFrame([profile.model_dump() for profile in payload.profiles])
        try:
            ranking, shortlist, warnings = score_profiles(frame, application.state.bundle, payload.k)
            monitoring = evaluate_batch(frame, application.state.reference, model_path=model_path)
        except ProfileValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=503, detail="The model and monitoring reference are inconsistent. Restore matching files and restart the API.") from error
        return {"model": application.state.bundle["metadata"]["selected_model"],
                "model_sha256": application.state.model_sha256, "schema_version": SCHEMA_VERSION,
                "count": len(ranking), "k": payload.k, "ranking": ranking.to_dict(orient="records"),
                "shortlist": shortlist.to_dict(orient="records"), "warnings": warnings,
                "monitoring": monitoring,
                "score_meaning": "Estimated probability of the synthetic Will_Buy_EV = Yes label; use for ranking, not a sales guarantee."}

    return application


app = create_app()
