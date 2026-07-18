"""
Préparation des données Zenodo (dt1 + dt2).
Code identique en local et sur Colab.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_merge(dt1_path: str, dt2_path: str, resample_freq: str = "5min") -> pd.DataFrame:
    """
    Charge dt1 (météo) et dt2 (électrique + Fault), fusionne sur le temps,
    rééchantillonne et filtre les heures de jour.
    """
    # TODO S2 : implémenter avec chunksize pour dt1 (2.2 Go)
    pass


def filter_daytime(df: pd.DataFrame, gti_threshold: float = 50.0) -> pd.DataFrame:
    """Filtre les lignes de jour (GTI > seuil)."""
    return df[df["gti"] > gti_threshold].copy()


def split_chronological(df: pd.DataFrame, train_ratio=0.8):
    """
    Split chronologique (jamais aléatoire).
    Retourne (train, test) triés par temps.
    """
    df = df.sort_values("time")
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def split_train_val_test(df: pd.DataFrame, ratios=(0.7, 0.1, 0.2)):
    """
    Split chronologique 70/10/20 pour Autoencoder et LSTM.
    Retourne (train, val, test).
    """
    df = df.sort_values("time")
    n = len(df)
    i1 = int(n * ratios[0])
    i2 = int(n * (ratios[0] + ratios[1]))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def fit_scaler(train_normal: pd.DataFrame, features: list) -> StandardScaler:
    """Fit le scaler sur train normal UNIQUEMENT."""
    scaler = StandardScaler()
    scaler.fit(train_normal[features])
    return scaler


def create_sequences(data: np.ndarray, seq_length: int = 12) -> np.ndarray:
    """
    Crée des séquences pour le LSTM.
    À appeler SÉPARÉMENT sur chaque partition (pas de chevauchement aux frontières).
    """
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)
