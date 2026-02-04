"""
Adversarial debiasing model for FairCredit AI.

Based on:
Zhang, Lemoine, Mitchell (2018)
"Mitigating Unwanted Biases with Adversarial Learning"
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------
class GradientReversal(torch.autograd.Function):
    """
    Reverses gradients during backpropagation.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_adv * grad_output, None


# ---------------------------------------------------------------------
# Adversarial Debiasing Model
# ---------------------------------------------------------------------
class AdversarialDebiasingModel(nn.Module):
    """
    Predictor + adversary neural network for debiasing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        lambda_adv: float = 1.0,
    ) -> None:
        super().__init__()

        self.lambda_adv = lambda_adv

        # Predictor network (creditworthiness)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Adversary network (predict sensitive attribute)
        self.adversary = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.bce = nn.BCELoss()

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self, X: torch.Tensor, sensitive_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            y_pred, s_pred
        """
        y_pred = self.predictor(X)

        # Gradient reversal
        y_reversed = GradientReversal.apply(y_pred, self.lambda_adv)

        s_pred = self.adversary(y_reversed)

        return y_pred, s_pred

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_attr: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> None:
        """
        Train the adversarial debiasing model.
        """
        logger.info("Training adversarial debiasing model")

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        s_t = torch.tensor(sensitive_attr.reshape(-1, 1), dtype=torch.float32)

        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            y_pred, s_pred = self.forward(X_t, s_t)

            loss_main = self.bce(y_pred, y_t)
            loss_adv = self.bce(s_pred, s_t)

            loss = loss_main - self.lambda_adv * loss_adv
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %d | Main loss: %.4f | Adv loss: %.4f",
                    epoch,
                    loss_main.item(),
                    loss_adv.item(),
                )

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            y_pred = self.predictor(X_t)
        return (y_pred.numpy().flatten() >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            y_pred = self.predictor(X_t)
        return y_pred.numpy().flatten()
