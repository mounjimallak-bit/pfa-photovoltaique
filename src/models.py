"""
Définitions des modèles de détection / prédiction.
"""
import torch
import torch.nn as nn


# --- Autoencoder simple ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# --- LSTM Autoencoder ---
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]  # (batch, hidden_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, seq_length: int, n_layers: int = 1):
        super().__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        out, _ = self.lstm(z)
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, seq_length: int = 12):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.decoder = LSTMDecoder(hidden_dim, input_dim, seq_length)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
