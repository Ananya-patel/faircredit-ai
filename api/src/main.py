"""
FairCredit API Service

Provides creditworthiness predictions with built-in
fairness safeguards and explainability hooks.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------
# Model Registry (in-memory for now)
# ---------------------------------------------------------------------
baseline_model: FairnessConstrainedClassifier | None = None
debiased_model: AdversarialDebiasingModel | None = None

# ---------------------------------------------------------------------
# Startup hook (NO SHAP HERE)
# ---------------------------------------------------------------------
@app.on_event("startup")
def load_models() -> None:
    """
    Initialize models.
    NOTE: Models are NOT trained here.
    """
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
    if baseline_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not hasattr(baseline_model.model, "coef_"):
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet",
        )

    X = pd.DataFrame([request.dict()])
    prob = float(baseline_model.model.predict_proba(X)[0, 1])
    approved = prob >= 0.5

    return CreditResponse(
        approved=approved,
        probability=prob,
        scoring_model="baseline_logistic",
    )


# ---------------------------------------------------------------------
# Prediction + Explainability (LAZY SHAP)
# ---------------------------------------------------------------------
@app.post("/predict_explain")
def predict_with_explanation(request: CreditRequest) -> Dict:
    if baseline_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not hasattr(baseline_model.model, "coef_"):
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet; explanations unavailable",
        )

    X = pd.DataFrame([request.dict()])
    prob = float(baseline_model.model.predict_proba(X)[0, 1])
    approved = prob >= 0.5

    explainer = ShapExplainer(baseline_model.model)
    explanation = explainer.explain(X)

    return {
        "approved": approved,
        "probability": prob,
        "scoring_model": "baseline_logistic",
        "explanation": explanation,
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
