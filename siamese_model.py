"""
Siamese Neural Network for pairwise ETF ranking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SiameseEncoder(nn.Module):
    """Shared encoder that processes individual ETF feature vectors."""
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        return self.net(x)


class SiameseComparator(nn.Module):
    """Comparator head that predicts P(i > j) from paired embeddings."""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        # Input: [E_i, E_j, ΔE, |ΔE|] = 4 * input_dim
        self.net = nn.Sequential(
            nn.Linear(4 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, e_i, e_j):
        delta = e_i - e_j
        abs_delta = torch.abs(delta)
        x = torch.cat([e_i, e_j, delta, abs_delta], dim=-1)
        return self.net(x).squeeze(-1)


class SiameseRanker:
    def __init__(self, input_dim, hidden_dims=None, lr=0.001, seed=42):
        if hidden_dims is None:
            hidden_dims = [64, 32]
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = SiameseEncoder(input_dim, hidden_dims).to(self.device)
        self.comparator = SiameseComparator(self.encoder.output_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.comparator.parameters()), lr=lr
        )
        self.criterion = nn.BCELoss()

    def fit(self, X1, X2, labels, epochs=100, batch_size=128):
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X1), torch.tensor(X2), torch.tensor(labels)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X1, batch_X2, batch_y in loader:
                batch_X1, batch_X2, batch_y = batch_X1.to(self.device), batch_X2.to(self.device), batch_y.to(self.device)
                
                e_i = self.encoder(batch_X1)
                e_j = self.encoder(batch_X2)
                preds = self.comparator(e_i, e_j)
                
                loss = self.criterion(preds, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(batch_X1)
            
            avg_loss = total_loss / len(X1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {
                    'encoder': {k: v.clone() for k, v in self.encoder.state_dict().items()},
                    'comparator': {k: v.clone() for k, v in self.comparator.state_dict().items()}
                }
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}")

        if best_state:
            self.encoder.load_state_dict(best_state['encoder'])
            self.comparator.load_state_dict(best_state['comparator'])

    def compute_conviction_scores(self, features_dict: dict) -> dict:
        """
        Compute conviction scores for all ETFs.
        features_dict: {ticker: feature_vector}
        Returns: {ticker: conviction_score}
        """
        tickers = list(features_dict.keys())
        n = len(tickers)
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        
        # Build feature matrix
        X = np.stack([features_dict[t] for t in tickers])
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.encoder.eval()
        self.comparator.eval()
        
        with torch.no_grad():
            embeddings = self.encoder(X_tensor)
            
            scores = {t: 0.0 for t in tickers}
            for i, t1 in enumerate(tickers):
                for j, t2 in enumerate(tickers):
                    if i == j:
                        continue
                    prob = self.comparator(
                        embeddings[i:i+1], embeddings[j:j+1]
                    ).item()
                    scores[t1] += prob
            
            # Average probability of outperforming other ETFs
            for t in tickers:
                scores[t] /= (n - 1)
        
        return scores
