"""
Chargement du dataset ML figé.

Les partitions et les scalers sont LUS depuis le disque, jamais recalculés :
c'est ce qui garantit que l'Isolation Forest, l'Autoencoder et le LSTM-AE
sont comparés sur exactement les mêmes données.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

RACINE   = Path(__file__).resolve().parents[1]
DIR_DATA = RACINE / "data" / "processed"
DIR_META = RACINE / "models" / "dataset"

_META = None


def manifeste() -> dict:
    global _META
    if _META is None:
        with open(DIR_META / "dataset_meta.json", encoding="utf-8") as f:
            _META = json.load(f)
    return _META


def features() -> list:
    """Les 14 features ML, source unique de vérité."""
    return list(manifeste()["features_ml"])


def charger_brut(nom: str) -> pd.DataFrame:
    """nom ∈ {train, val, test, train_normal} → DataFrame indexé par temps."""
    m = manifeste()
    if m["format"] == "parquet":
        df = pd.read_parquet(DIR_DATA / f"{nom}.parquet")
    else:
        df = pd.read_csv(DIR_DATA / f"{nom}.csv", index_col="time")
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def scaler(profil: str = "Novelty"):
    """Scaler figé. Ne JAMAIS appeler .fit() dessus."""
    return joblib.load(RACINE / manifeste()["scalers"][profil])


def charger(nom: str, profil: str = "Novelty"):
    """Retourne (X normalisé, y binaire, f types de panne, index temporel)."""
    df = charger_brut(nom)
    X  = scaler(profil).transform(df[features()])
    y  = (df["Fault"] != 0).astype(int).values
    return X, y, df["Fault"].values, df.index


def segments_continus(index, pas: str = "5min", tol: float = 1.5):
    """
    Découpe un index temporel en blocs contigus.

    La série ne contient que les heures de jour : entre le coucher et le lever
    du soleil il y a un trou de plusieurs heures. Construire des séquences LSTM
    sans en tenir compte fabriquerait des transitions soir → matin inexistantes.
    Retourne une liste de (debut, fin) en positions entières.
    """
    dt       = pd.Timedelta(pas)
    ecarts   = index.to_series().diff()
    ruptures = np.flatnonzero((ecarts > dt * tol).values)
    bornes   = [0, *ruptures.tolist(), len(index)]
    return [(a, b) for a, b in zip(bornes[:-1], bornes[1:]) if b > a]


def sequences(X, longueur: int = 12, index=None, pas: str = "5min"):
    """
    Fenêtres glissantes de `longueur` pas, sans jamais franchir un trou.
    Retourne (séquences, positions de fin) — les positions servent à réaligner
    y et Fault sur les séquences, qui sont moins nombreuses que les points.
    """
    blocs = [(0, len(X))] if index is None else segments_continus(index, pas)
    seqs, fins = [], []
    for a, b in blocs:
        for i in range(a, b - longueur + 1):
            seqs.append(X[i:i + longueur])
            fins.append(i + longueur - 1)
    return np.asarray(seqs), np.asarray(fins)

def index_evaluable(nom: str, longueur: int = 12, pas: str = "5min"):
    """
    Timestamps qu'un modèle séquentiel de `longueur` pas peut réellement scorer.
    Base commune pour comparer équitablement IF, AE et LSTM.
    """
    df = charger_brut(nom)
    _, fins = sequences(df[features()].values, longueur=longueur,
                        index=df.index, pas=pas)
    return df.index[fins]