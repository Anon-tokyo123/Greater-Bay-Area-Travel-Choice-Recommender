from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Assume your data loading functions from vtts_model.py are imported here ---
from model import load_and_clean_data, create_long_format, calculate_vtts_table, X_cols

app = FastAPI(title="GBA Travel Mode & VTTS API", version="1.0")

# --- 1. Global Setup & Model Training ---
print("Initializing API, loading dataset, and training models...")
survey_data, attr_dicts, _ = load_and_clean_data()
df_long_all = create_long_format(survey_data, attr_dicts)

# We need the full feature list for prediction
asc_features = X_cols + ['ASC_HSR', 'ASC_Taxi', 'ASC_PrivateCar', 'ASC_eVTOL']

# We will pre-train models for specific scenarios so the API responds instantly.
# Let's create a dictionary to hold our trained models and scalers.
models_cache = {}

def train_scenario_model(purpose, distance):
    subset = df_long_all[(df_long_all['Purpose'] == purpose) & (df_long_all['Distance_Category'] == distance)]
    if subset.empty: return None, None
    
    X = subset[asc_features]
    y = subset['is_chosen']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(penalty='l1', solver='saga', C=5.0, max_iter=5000, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# Pre-train a model for "Work" + "Long (150km)" as an example
models_cache[("Work", "Long (150km)")] = train_scenario_model("Work", "Long (150km)")
print("API Ready.")

# --- 2. Defining Enums and Schemas ---

# This Enum creates the "Dropdown Menu" (Button equivalent) in FastAPI /docs
class TripPurpose(str, Enum):
    work = "Work"
    non_work = "Non-Work"

class DistanceCategory(str, Enum):
    short = "Short (50km)"
    medium = "Medium (100km)"
    long = "Long (150km)"

class ModeChoice(str, Enum):
    hsr = "HSR"
    taxi = "Taxi"
    private_car = "Private Car"
    evtol = "eVTOL"

# This defines the data structure the user must send to the prediction endpoint
class TripParameters(BaseModel):
    fare: float
    in_vehicle_time: float
    waiting_time: float
    access_egress_time: float
    transfer_time: float
    crowding_level: float # e.g., 0.5 for 50%
    customs_clearance_time: float

# --- 3. Endpoints ---

@app.get("/")
def root():
    return {"message": "Welcome to the VTTS API. Go to /docs to test the endpoints."}

@app.get("/vtts")
def get_vtts_by_purpose(purpose: TripPurpose):
    """
    Returns the VTTS based on trip purpose. The 'Purpose' input is a dropdown.
    """
    df_subset = df_long_all[df_long_all['Purpose'] == purpose.value].copy()
    
    if df_subset.empty:
        raise HTTPException(status_code=404, detail="No data found for this purpose.")

    vtts_df = calculate_vtts_table(df_subset)
    
    return {
        "trip_purpose": purpose.value,
        "vtts_hkd_per_hour": vtts_df.to_dict()
    }

@app.post("/predict_choice")
def predict_mode_probability(
    purpose: TripPurpose,
    distance: DistanceCategory,
    mode: ModeChoice,
    params: TripParameters
):
    """
    Predicts the probability that a user will choose a specific mode of transport 
    given custom travel parameters (fare, time, crowding).
    """
    model_data = models_cache.get((purpose.value, distance.value))
    
    if not model_data or model_data[0] is None:
        # If we haven't pre-trained it, train it right now
        model, scaler = train_scenario_model(purpose.value, distance.value)
        if model is None:
             raise HTTPException(status_code=404, detail="Not enough data to train this scenario.")
        models_cache[(purpose.value, distance.value)] = (model, scaler)
    else:
        model, scaler = model_data

    # Set up the ASC dummy variables based on the requested mode
    asc_hsr = 1 if mode == ModeChoice.hsr else 0
    asc_taxi = 1 if mode == ModeChoice.taxi else 0
    asc_private = 1 if mode == ModeChoice.private_car else 0
    asc_evtol = 1 if mode == ModeChoice.evtol else 0

    # Format the input data to match the training data exactly
    input_data = pd.DataFrame([{
        'fare': params.fare,
        'in-vehicle time': params.in_vehicle_time,
        'waiting time': params.waiting_time,
        'access & egress time': params.access_egress_time,
        'transfer time': params.transfer_time,
        'crowding level': params.crowding_level,
        'customs clearance time': params.customs_clearance_time,
        'ASC_HSR': asc_hsr,
        'ASC_Taxi': asc_taxi,
        'ASC_PrivateCar': asc_private,
        'ASC_eVTOL': asc_evtol
    }])

    # Scale the custom parameters using the saved scaler
    input_scaled = scaler.transform(input_data)
    
    # Predict the probability (index 1 is the probability of "is_chosen" = 1)
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "scenario": f"{purpose.value} - {distance.value}",
        "tested_mode": mode.value,
        "input_parameters": params.dict(),
        "probability_of_choosing": round(probability, 4)
    }
