"""
FairCredit API Service

Provides creditworthiness predictions with built-in
fairness safeguards and explainability hooks.
"""

from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware


import logging
from datetime import datetime
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.fair_classifiers import FairnessConstrainedClassifier
from src.models.adversarial_debiasing import AdversarialDebiasingModel
from src.evaluation.explainability import ShapExplainer

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faircredit-api")

# ---------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------
app = FastAPI(
    title="FairCredit API",
    description="Ethical alternative credit scoring system",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------
class CreditRequest(BaseModel):
    payment_regularity: float = Field(..., ge=0.0, le=1.0)
    income_consistency: float = Field(..., ge=0.0, le=1.0)
    emergency_fund_ratio: float = Field(..., ge=0.0, le=1.0)
    network_creditworthiness: float = Field(..., ge=0.0, le=1.0)
    community_involvement: float = Field(..., ge=0.0, le=1.0)


class CreditResponse(BaseModel):
    approved: bool
    probability: float
    scoring_model: str


class SimulatorRequest(BaseModel):
    savings_rate: float = Field(..., ge=0, le=1)
    payment_regularity: float = Field(..., ge=0, le=1)
    transaction_consistency: float = Field(..., ge=0, le=1)
    employment_stability: int = Field(..., ge=0, le=50)
    network_diversity: float = Field(..., ge=0, le=1)


# ---------------------------------------------------------------------
# Helper: Simulator → Model feature adapter
# ---------------------------------------------------------------------
def simulator_to_model_features(sim_input: SimulatorRequest) -> Dict[str, float]:
    """
    Convert simulator behavioral inputs into model-ready features.
    IMPORTANT: Must match trained model feature schema exactly.
    """
    return {
        "payment_regularity": sim_input.payment_regularity,
        "income_consistency": sim_input.transaction_consistency,
        "emergency_fund_ratio": min(sim_input.savings_rate * 2, 1.0),
        "network_creditworthiness": sim_input.network_diversity,
        "community_involvement": sim_input.payment_regularity,
    }


# ---------------------------------------------------------------------
# Model Registry (in-memory)
# ---------------------------------------------------------------------
baseline_model: FairnessConstrainedClassifier | None = None
debiased_model: AdversarialDebiasingModel | None = None

# ---------------------------------------------------------------------
# Startup hook
# ---------------------------------------------------------------------
@app.on_event("startup")
def load_models() -> None:
    global baseline_model, debiased_model

    logger.info("Initializing models for API")

    baseline_model = FairnessConstrainedClassifier()
    debiased_model = AdversarialDebiasingModel(input_dim=5)

    logger.info("Models initialized (not yet trained)")


# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------
@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest) -> CreditResponse:
    if baseline_model is None or not hasattr(baseline_model.model, "coef_"):
        raise HTTPException(status_code=503, detail="Model not trained")

    X = pd.DataFrame([request.dict()])
    prob = float(baseline_model.model.predict_proba(X)[0, 1])

    return CreditResponse(
        approved=prob >= 0.5,
        probability=prob,
        scoring_model="baseline_logistic",
    )


# ---------------------------------------------------------------------
# Prediction + Explainability
# ---------------------------------------------------------------------
@app.post("/predict_explain")
def predict_with_explanation(request: CreditRequest) -> Dict:
    if baseline_model is None or not hasattr(baseline_model.model, "coef_"):
        raise HTTPException(status_code=503, detail="Model not trained")

    X = pd.DataFrame([request.dict()])
    prob = float(baseline_model.model.predict_proba(X)[0, 1])

    explainer = ShapExplainer(baseline_model.model)
    explanation = explainer.explain(X)

    return {
        "approved": prob >= 0.5,
        "probability": prob,
        "scoring_model": "baseline_logistic",
        "explanation": explanation,
    }


# ---------------------------------------------------------------------
# Simulator Endpoint (PUBLIC DEMO)
# ---------------------------------------------------------------------
@app.post("/simulate")
def simulate_credit(request: SimulatorRequest) -> Dict:
    if baseline_model is None or not hasattr(baseline_model.model, "coef_"):
        raise HTTPException(status_code=503, detail="Model not trained")

    features = simulator_to_model_features(request)
    X = pd.DataFrame([features])

    prob = float(baseline_model.model.predict_proba(X)[0, 1])
    decision = "Approved" if prob >= 0.5 else "Declined"

    explainer = ShapExplainer(baseline_model.model)
    explanation = explainer.explain(X)

    return {
        "decision": decision,
        "risk_score": round(1 - prob, 3),
        "explanations": explanation,
        "fairness_note": "No protected attributes were used.",
    }


# ---------------------------------------------------------------------
# Regulatory Audit Endpoint
# ---------------------------------------------------------------------
@app.get("/audit")
def audit() -> Dict:
    return {
        "model_name": "FairCredit Baseline Logistic",
        "version": "1.0.0",
        "status": "untrained-demo",
        "features_used": [
            "payment_regularity",
            "income_consistency",
            "emergency_fund_ratio",
            "network_creditworthiness",
            "community_involvement",
        ],
        "fairness_constraints": [
            "demographic_parity",
            "equal_opportunity",
        ],
        "generated_at": datetime.utcnow().isoformat(),
    }
