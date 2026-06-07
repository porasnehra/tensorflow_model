import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import tensorflow as tf

from models.tf_model import load_tf_model
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
        print("Warning: Model file not found. Please train the TensorFlow model first.")
except Exception as e:
    print(f"Error loading model: {e}")

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
