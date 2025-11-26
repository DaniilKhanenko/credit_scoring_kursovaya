import joblib
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

XGB_PATH = Path("models/xgb_model.pkl")
NN_PREP_PATH = Path("models/nn_preprocessor.pkl")
NN_MODEL_PATH = Path("models/nn_model.pt")

_xgb_data = None
_nn_preproc = None
_nn_model = None

class LoanNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def get_neutral_score():
    global _xgb_data
    if _xgb_data is None:
        _xgb_data = joblib.load(XGB_PATH)
    return _xgb_data.get("neutral_score", 650)

def get_feature_cols():
    global _xgb_data
    if _xgb_data is None:
        _xgb_data = joblib.load(XGB_PATH)
    return _xgb_data["feature_cols"]

def predict_xgb(input_dict):
    global _xgb_data
    if _xgb_data is None:
        _xgb_data = joblib.load(XGB_PATH)
    
    pipeline = _xgb_data["pipeline"]
    cols = _xgb_data["feature_cols"]
    
    df = pd.DataFrame([input_dict], columns=cols)
    

    proba = pipeline.predict_proba(df)[0, 1]
    return float(proba)

def predict_nn(input_dict):
    global _nn_preproc, _nn_model
    
    if _nn_preproc is None:
        _nn_preproc = joblib.load(NN_PREP_PATH)
    
    if _nn_model is None:
        checkpoint = torch.load(NN_MODEL_PATH)
        input_dim = checkpoint["input_dim"]
        model = LoanNN(input_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _nn_model = model
        
    cols = get_feature_cols()
    df = pd.DataFrame([input_dict], columns=cols)
    
    X_transformed = _nn_preproc.transform(df)
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
    
    with torch.no_grad():
        proba = _nn_model(X_tensor).item()
    
    return float(proba)

def predict_approval_proba(input_dict, model_name="xgb"):
    if model_name == "xgb":
        return predict_xgb(input_dict)
    elif model_name == "nn":
        return predict_nn(input_dict)
    else:
        return 0.0
