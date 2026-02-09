# FairCredit — Ethical Credit Scoring Platform

FairCredit is a production-grade machine learning system that predicts
creditworthiness using alternative financial data while enforcing
fairness, transparency, and regulatory compliance.

---

##  Live Deployment

The FairCredit API is publicly deployed and accessible:

- **Swagger API Docs:** https://faircredit-ai.onrender.com/docs
- **Health Check:** https://faircredit-ai.onrender.com/health

The service is containerized using Docker and deployed on Render.

---

##  Key Features

- Alternative data–based credit scoring
- Fairness-aware modeling with bias mitigation
- Explainability using SHAP
- Regulatory audit endpoints
- FastAPI-based production API
- Dockerized deployment
- Monitoring and retraining pipeline

---

##  Tech Stack

- Python 3.10
- FastAPI
- scikit-learn, PyTorch
- SHAP (Explainable AI)
- Docker & Docker Compose
- Streamlit (Monitoring)
- Render (Deployment)

---

##  Run Locally

```bash
docker compose up --build
## live link
https://faircredit-ai-3.onrender.com/