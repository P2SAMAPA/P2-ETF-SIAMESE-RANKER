"""
siamese_model.py
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


class Encoder(nn.Module):
    """Shared encoder — same weights applied to both ETFs in a pair."""

    def __init__(self, input_dim: int, hidden_dims: list = [64, 32], dropout: float = 0.1):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.net        = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x):
        return self.net(x)


class SiameseRanker(nn.Module):
    """
    Full Siamese ranking model.
    concat([E_i, E_j, E_i-E_j, |E_i-E_j|]) → Dense head → P(i > j)
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list = [64, 32],
        head_dims:   list = [32, 16],
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.encoder    = Encoder(input_dim, hidden_dims, dropout)
        emb_dim         = self.encoder.output_dim
        comparator_dim  = 4 * emb_dim

        head_layers, in_dim = [], comparator_dim
        for h in head_dims:
            head_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        head_layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.head = nn.Sequential(*head_layers)

    def forward(self, xi, xj):
        ei    = self.encoder(xi)
        ej    = self.encoder(xj)
        delta = ei - ej
        return self.head(torch.cat([ei, ej, delta, delta.abs()], dim=-1)).squeeze(-1)

    def predict_proba(self, xi: np.ndarray, xj: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self.forward(
                torch.tensor(xi, dtype=torch.float32).to(device),
                torch.tensor(xj, dtype=torch.float32).to(device),
            ).cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_siamese(
    Xi_train, Xj_train, y_train,
    Xi_val,   Xj_val,   y_val,
    input_dim:               int,
    hidden_dims:             list  = [64, 32],
    head_dims:               list  = [32, 16],
    dropout:                 float = 0.1,
    epochs:                  int   = 20,
    batch_size:              int   = 256,
    lr:                      float = 0.001,
    early_stopping_patience: int   = 5,
    device:                  str   = "cpu",
    time_limit_seconds:      Optional[float] = None,
) -> Tuple[SiameseRanker, dict]:

    model     = SiameseRanker(input_dim, hidden_dims, head_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    def make_loader(Xi, Xj, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(Xi, dtype=torch.float32),
            torch.tensor(Xj, dtype=torch.float32),
            torch.tensor(y,  dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(Xi_train, Xj_train, y_train, shuffle=True)
    val_loader   = make_loader(Xi_val,   Xj_val,   y_val)

    history          = {"train_loss": [], "val_loss": []}
    best_val_loss    = float("inf")
    best_state       = None
    patience_counter = 0
    start_time       = time.time()

    for epoch in range(epochs):
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            logger.warning(f"Time limit {time_limit_seconds}s hit at epoch {epoch}.")
            break

        model.train()
        train_losses = []
        for xi_b, xj_b, y_b in train_loader:
            xi_b, xj_b, y_b = xi_b.to(device), xj_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xi_b, xj_b), y_b)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xi_b, xj_b, y_b in val_loader:
                xi_b, xj_b, y_b = xi_b.to(device), xj_b.to(device), y_b.to(device)
                val_losses.append(criterion(model(xi_b, xj_b), y_b).item())

        tl, vl = np.mean(train_losses), np.mean(val_losses)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train: {tl:.4f} | Val: {vl:.4f}")

        if vl < best_val_loss:
            best_val_loss    = vl
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    history["elapsed_seconds"] = time.time() - start_time
    logger.info(f"Done in {history['elapsed_seconds']:.1f}s. Best val loss: {best_val_loss:.4f}")
    return model, history


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(model: SiameseRanker, path: str, meta: dict = None):
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)
    logger.info(f"Model saved → {path}")


def load_model(path: str, input_dim: int, hidden_dims: list, head_dims: list) -> SiameseRanker:
    payload = torch.load(path, map_location="cpu")
    model   = SiameseRanker(input_dim, hidden_dims, head_dims)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
