"""
core/siamese_model.py
Siamese neural network for pairwise ETF ranking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import time
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Shared encoder — same weights applied to both ETFs in a pair."""

    def __init__(self, input_dim: int, hidden_dims: list = [64, 32], dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Siamese Comparator + Head
# ---------------------------------------------------------------------------

class SiameseRanker(nn.Module):
    """
    Full Siamese ranking model.

    Architecture:
        Encoder(Xi) → E_i
        Encoder(Xj) → E_j  (shared weights)
        Comparator: concat([E_i, E_j, E_i - E_j, |E_i - E_j|])
        Head: Dense → Dense → Sigmoid → P(i > j)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        head_dims: list = [32, 16],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, dropout)
        emb_dim = self.encoder.output_dim

        # Comparator input: [E_i, E_j, ΔE, |ΔE|] = 4 * emb_dim
        comparator_dim = 4 * emb_dim

        head_layers = []
        in_dim = comparator_dim
        for h in head_dims:
            head_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        head_layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.head = nn.Sequential(*head_layers)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        ei = self.encoder(xi)
        ej = self.encoder(xj)
        delta = ei - ej
        comparator_input = torch.cat([ei, ej, delta, delta.abs()], dim=-1)
        return self.head(comparator_input).squeeze(-1)

    def predict_proba(self, xi: np.ndarray, xj: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Predict P(i > j) for numpy arrays."""
        self.eval()
        with torch.no_grad():
            xi_t = torch.tensor(xi, dtype=torch.float32).to(device)
            xj_t = torch.tensor(xj, dtype=torch.float32).to(device)
            return self.forward(xi_t, xj_t).cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_siamese(
    Xi_train: np.ndarray,
    Xj_train: np.ndarray,
    y_train: np.ndarray,
    Xi_val: np.ndarray,
    Xj_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dims: list = [64, 32],
    head_dims: list = [32, 16],
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.001,
    early_stopping_patience: int = 5,
    device: str = "cpu",
    time_limit_seconds: Optional[float] = None,
) -> Tuple[SiameseRanker, dict]:
    """
    Train Siamese ranker with BCE loss.

    Returns:
        model: Trained SiameseRanker
        history: dict with train/val loss per epoch
    """
    model = SiameseRanker(input_dim, hidden_dims, head_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(Xi_train, dtype=torch.float32),
        torch.tensor(Xj_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(Xi_val, dtype=torch.float32),
        torch.tensor(Xj_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        # Check time limit
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            logger.warning(f"Time limit {time_limit_seconds}s reached at epoch {epoch}. Stopping.")
            break

        # Train
        model.train()
        train_losses = []
        for xi_b, xj_b, y_b in train_loader:
            xi_b, xj_b, y_b = xi_b.to(device), xj_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(xi_b, xj_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xi_b, xj_b, y_b in val_loader:
                xi_b, xj_b, y_b = xi_b.to(device), xj_b.to(device), y_b.to(device)
                pred = model(xi_b, xj_b)
                loss = criterion(pred, y_b)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - start_time
    history["elapsed_seconds"] = elapsed
    logger.info(f"Training complete in {elapsed:.1f}s. Best val loss: {best_val_loss:.4f}")

    return model, history


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(model: SiameseRanker, path: str, meta: dict = None):
    """Save model state dict + metadata."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict(), "meta": meta or {}}
    torch.save(payload, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str, input_dim: int, hidden_dims: list, head_dims: list) -> SiameseRanker:
    """Load model from state dict."""
    payload = torch.load(path, map_location="cpu")
    model = SiameseRanker(input_dim, hidden_dims, head_dims)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
