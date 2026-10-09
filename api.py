"""FastAPI wrapper for the private-data GBA travel choice model."""

from enum import Enum
from functools import lru_cache
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model import X_cols, calculate_vtts_table, create_long_format, load_and_clean_data

app = FastAPI(
    title="GBA Travel Mode & VTTS API",
    version="1.1",
    description=(
        "Academic prototype. Use GBA_DEMO_MODE=1 for synthetic demonstration data "
        "or supply the restricted survey workbook locally. Demo outputs are not "
        "research results; binary scores are not normalized across travel modes."
    ),
)

ASC_FEATURES = X_cols + ["ASC_HSR", "ASC_Taxi", "ASC_PrivateCar", "ASC_eVTOL"]
models_cache = {}
DEMO_MODE = os.getenv("GBA_DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
DEMO_NOTICE = "Synthetic demo only; these are not project survey results."


def create_demo_long_data(seed=42, choice_sets_per_scenario=300):
    """Create deterministic synthetic choice data for an API-only demo.

    These generated records do not contain or approximate the restricted
    survey responses. They only exercise the feature pipeline and endpoints.
    """
    rng = np.random.default_rng(seed)
    modes = [
        (1, "Bus/MTR"), (2, "HSR"), (3, "Taxi"),
        (4, "Private Car"), (5, "eVTOL"),
    ]
    mode_asc = {
        "Bus/MTR": (0.0, "baseline"),
        "HSR": (0.35, "ASC_HSR"),
        "Taxi": (-0.55, "ASC_Taxi"),
        "Private Car": (-0.25, "ASC_PrivateCar"),
        "eVTOL": (-0.8, "ASC_eVTOL"),
    }
    distance_factors = {
        "Short (50km)": 1.0,
        "Medium (100km)": 1.8,
        "Long (150km)": 2.5,
    }
    records = []

    for purpose in ("Work", "Non-Work"):
        time_sensitivity = 0.020 if purpose == "Work" else 0.014
        for distance, distance_factor in distance_factors.items():
            for _ in range(choice_sets_per_scenario):
                alternatives = []
                utilities = []
                for mode_idx, mode_name in modes:
                    fare_bases = {1: 45, 2: 120, 3: 700, 4: 300, 5: 600}
                    time_bases = {1: 95, 2: 45, 3: 75, 4: 85, 5: 28}
                    fare = max(5.0, rng.normal(fare_bases[mode_idx] * distance_factor, 12.0))
                    in_vehicle = max(5.0, rng.normal(time_bases[mode_idx] * distance_factor, 9.0))
                    waiting = max(0.0, rng.normal({1: 8, 2: 15, 3: 5, 4: 0, 5: 10}[mode_idx], 3.0))
                    access_egress = max(0.0, rng.normal({1: 14, 2: 18, 3: 8, 4: 10, 5: 12}[mode_idx], 4.0))
                    transfer = max(0.0, rng.normal({1: 12, 2: 4, 3: 0, 4: 0, 5: 3}[mode_idx], 3.0))
                    crowding = float(np.clip(rng.normal({1: 0.65, 2: 0.3, 3: 0.1, 4: 0.0, 5: 0.05}[mode_idx], 0.12), 0, 1))
                    customs = max(0.0, rng.normal(15 if mode_idx in (1, 2, 5) else 10, 4.0))
                    asc, _ = mode_asc[mode_name]
                    utility = (
                        -0.006 * fare - time_sensitivity * in_vehicle
                        -0.012 * waiting -0.009 * access_egress
                        -0.010 * transfer -0.55 * crowding -0.014 * customs + asc
                    )
                    alternatives.append({
                        "fare": fare,
                        "in-vehicle time": in_vehicle,
                        "waiting time": waiting,
                        "access & egress time": access_egress,
                        "transfer time": transfer,
                        "crowding level": crowding,
                        "customs clearance time": customs,
                        "ASC_HSR": int(mode_name == "HSR"),
                        "ASC_Taxi": int(mode_name == "Taxi"),
                        "ASC_PrivateCar": int(mode_name == "Private Car"),
                        "ASC_eVTOL": int(mode_name == "eVTOL"),
                        "Purpose": purpose,
                        "Distance_Category": distance,
                    })
                    utilities.append(utility)

                shifted = np.asarray(utilities) - max(utilities)
                probabilities = np.exp(shifted) / np.exp(shifted).sum()
                chosen_idx = int(rng.choice(len(modes), p=probabilities))
                for index, alternative in enumerate(alternatives):
                    records.append({**alternative, "is_chosen": int(index == chosen_idx)})

    return pd.DataFrame.from_records(records)


@lru_cache(maxsize=1)
def get_long_data():
    """Load synthetic or private survey data only when an endpoint needs it."""
    if DEMO_MODE:
        return create_demo_long_data()
    survey_data, attr_dicts, _ = load_and_clean_data()
    return create_long_format(survey_data, attr_dicts)


def get_data_source():
    return "synthetic_demo" if DEMO_MODE else "restricted_survey"


def get_source_metadata():
    metadata = {"data_source": get_data_source()}
    if DEMO_MODE:
        metadata["notice"] = DEMO_NOTICE
    return metadata


def require_long_data():
    try:
        data = get_long_data()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if data.empty:
        raise HTTPException(status_code=503, detail="No usable survey records were loaded.")
    return data


def train_scenario_model(purpose, distance, data):
    subset = data[(data["Purpose"] == purpose) & (data["Distance_Category"] == distance)]
    if subset.empty or subset["is_chosen"].nunique() < 2:
        return None, None

    missing = [column for column in ASC_FEATURES if column not in subset.columns]
    if missing:
        raise ValueError(f"The loaded data is missing required columns: {', '.join(missing)}")

    X = subset[ASC_FEATURES]
    y = subset["is_chosen"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        solver="saga", l1_ratio=1.0, C=5.0, max_iter=5000, random_state=42
    )
    model.fit(X_scaled, y)
    return model, scaler


class TripPurpose(str, Enum):
    work = "Work"
    non_work = "Non-Work"


class DistanceCategory(str, Enum):
    short = "Short (50km)"
    medium = "Medium (100km)"
    long = "Long (150km)"


class ModeChoice(str, Enum):
    bus_mtr = "Bus/MTR"
    hsr = "HSR"
    taxi = "Taxi"
    private_car = "Private Car"
    evtol = "eVTOL"


class TripParameters(BaseModel):
    fare: float = Field(ge=0, description="Fare in HKD")
    in_vehicle_time: float = Field(ge=0, description="In-vehicle minutes")
    waiting_time: float = Field(ge=0, description="Waiting minutes")
    access_egress_time: float = Field(ge=0, description="Access and egress minutes")
    transfer_time: float = Field(ge=0, description="Transfer minutes")
    crowding_level: float = Field(ge=0, le=1, description="Crowding proportion from 0 to 1")
    customs_clearance_time: float = Field(ge=0, description="Customs clearance minutes")


@app.get("/")
def root():
    return {
        "message": "GBA travel choice prototype. Open /docs for the API.",
        **get_source_metadata(),
    }


@app.get("/vtts")
def get_vtts_by_purpose(purpose: TripPurpose):
    data = require_long_data()
    subset = data[data["Purpose"] == purpose.value].copy()
    if subset.empty:
        raise HTTPException(status_code=404, detail="No data found for this trip purpose.")
    try:
        vtts_df = calculate_vtts_table(subset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "trip_purpose": purpose.value,
        **get_source_metadata(),
        "vtts_hkd_per_hour": vtts_df.to_dict(),
    }


@app.post("/predict_choice")
def predict_mode_score(
    purpose: TripPurpose,
    distance: DistanceCategory,
    mode: ModeChoice,
    params: TripParameters,
):
    """Return the binary model score for a single requested mode."""
    data = require_long_data()
    key = (purpose.value, distance.value)
    if key not in models_cache:
        try:
            models_cache[key] = train_scenario_model(*key, data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    model, scaler = models_cache[key]
    if model is None:
        raise HTTPException(
            status_code=404,
            detail="Insufficient data or only one choice class for this scenario.",
        )

    mode_features = {
        "ASC_HSR": int(mode == ModeChoice.hsr),
        "ASC_Taxi": int(mode == ModeChoice.taxi),
        "ASC_PrivateCar": int(mode == ModeChoice.private_car),
        "ASC_eVTOL": int(mode == ModeChoice.evtol),
    }
    input_data = pd.DataFrame([{
        "fare": params.fare,
        "in-vehicle time": params.in_vehicle_time,
        "waiting time": params.waiting_time,
        "access & egress time": params.access_egress_time,
        "transfer time": params.transfer_time,
        "crowding level": params.crowding_level,
        "customs clearance time": params.customs_clearance_time,
        **mode_features,
    }], columns=ASC_FEATURES)
    probability = float(model.predict_proba(scaler.transform(input_data))[0][1])
    params_dict = params.model_dump() if hasattr(params, "model_dump") else params.dict()

    return {
        "scenario": f"{purpose.value} - {distance.value}",
        **get_source_metadata(),
        "tested_mode": mode.value,
        "input_parameters": params_dict,
        "estimated_binary_choice_probability": round(probability, 4),
        "interpretation": (
            "Binary chosen-vs-not-chosen model output for this mode; "
            "scores across modes are not normalized and may not sum to 1."
        ),
    }
