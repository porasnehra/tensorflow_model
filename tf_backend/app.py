import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import tensorflow as tf

import os
from models.tf_model import load_tf_model, create_tf_model
from utils.feature_engineering import engineer_features, get_feature_names
from predict import get_risk_tier, explain_signals
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mule Account Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing/development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load the TensorFlow global model
MODEL_PATH = Path("results/final_model.keras")
model = None

try:
    if MODEL_PATH.exists():
        model = load_tf_model(str(MODEL_PATH))
        print(f"Loaded TensorFlow model from {MODEL_PATH}")
    else:
        print("Warning: Model file not found. Generating an initial untrained model to prevent 503 errors.")
        os.makedirs("results", exist_ok=True)
        model = create_tf_model()
        model.save(str(MODEL_PATH))
        print(f"Untrained baseline model saved to {MODEL_PATH}")
except Exception as e:
    print(f"Error loading/creating model: {e}")

class PredictionRequest(BaseModel):
    # This accepts the raw JSON data sent by your Flutter app
    features: Dict[str, Any]

@app.get("/")
def read_root():
    return {"status": "Active", "message": "Mule Detection API is running. Send POST request to /predict"}

@app.post("/predict")
def predict_risk(request: PredictionRequest):
    try:
        # 1. Convert the incoming JSON dictionary from Flutter into a Pandas DataFrame (1 row)
        df = pd.DataFrame([request.features])
        
        # Automatically fill in any missing base columns the Flutter app forgot to send
        base_cols = ['F3799', 'F3800', 'F3796', 'F3797', 'F3922', 'F3890', 'F3895', 'F3894', 'F3891', 'F3919', 'F13', 'F14', 'F15', 'F16', 'F19', 'F25', 'F3856', 'F3859', 'F3882', 'F3883', 'F3905', 'F3900', 'F3920', 'F3923', 'F3915', 'F3912', 'F3901', 'F3902', 'F3889', 'F3886', 'F3893', 'F3887', 'F2796', 'F3877']
        for col in base_cols:
            if col not in df.columns:
                df[col] = 0
                
        # Fix string-based defaults
        if 'F3891' not in df.columns or pd.isna(df['F3891'].iloc[0]): df['F3891'] = 'others'
        if 'F3890' not in df.columns or pd.isna(df['F3890'].iloc[0]): df['F3890'] = 'U'
        if 'F3889' not in df.columns or pd.isna(df['F3889'].iloc[0]): df['F3889'] = 'G365D'
        if 'F3886' not in df.columns or pd.isna(df['F3886'].iloc[0]): df['F3886'] = 'Savings'
        if 'F3893' not in df.columns or pd.isna(df['F3893'].iloc[0]): df['F3893'] = 'RETAIL'
        
        # 2. Run your existing Feature Engineering pipeline
        features_df = engineer_features(df)
        feature_cols = get_feature_names()
        
        # Ensure all required features exist even if Flutter left some out (fill with 0)
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
                
        X = features_df[feature_cols].values.astype(np.float32)
        
        # 3. Predict the Mule Probability (TensorFlow syntax)
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded. Please train first.")
            
        prob = float(model.predict(X, verbose=0)[0][0])
        tier = get_risk_tier(prob)
        signals = explain_signals(features_df.iloc[0])
        
        # 4. Disabled SHAP Proofs (temporarily disabled for TF transition)
        proofs = []
        
        # 5. Return the result back to Flutter
        return {
            "mule_probability": round(prob, 4),
            "risk_tier": tier,
            "flagged": prob >= 0.35, # default threshold
            "signals_triggered": len(signals),
            "signals": signals,
            "proofs": proofs
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    # Render provides the PORT environment variable. If not found, default to 10000.
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
