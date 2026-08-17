"""
Système de détection d'anomalies PV : ensemble AE + LSTM, fusion par rangs.

Copie déployable de la classe définie en § 14 de notebooks/final.ipynb. Le
notebook reste la source de vérité : toute modification du modèle passe par lui,
ce fichier n'en est que le portage. La § 15 vérifie d'ailleurs que ce chemin
reproduit exactement le score de la § 12 (écart max 0.00e+00).
"""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from tensorflow import keras


RACINE = Path(__file__).resolve().parents[1]


class DetecteurPV:
    """Charge model_final/ et classe de nouvelles mesures."""

    def __init__(self, dossier=None, verbeux=True):
        dossier = Path(dossier) if dossier else RACINE / "model_final"
        self.dossier = dossier

        with open(dossier / "config.json", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.features = self.cfg["features"]
        self.seq_len = self.cfg["lstm"]["seq_len"]
        self.seuil = self.cfg["seuil_fusion"]
        self.gap_min = self.cfg["nuit_gap_minutes"]
        self._nf = len(self.features)

        self.aes = [keras.models.load_model(dossier / "ae" / f"ae_{s}.keras")
                    for s in range(self.cfg["ae"]["n_ensemble"])]
        self.lstms = [keras.models.load_model(dossier / "lstm" / f"lstm_{s}.keras")
                      for s in range(self.cfg["lstm"]["n_ensemble"])]

        with open(dossier / "scaler_ae.pkl", "rb") as f:
            self.sc_ae = pickle.load(f)
        with open(dossier / "scaler_lstm.pkl", "rb") as f:
            self.sc_lstm = pickle.load(f)

        # référence (scores val bruts) pour le rang — triée UNE fois
        self.ref_ae = np.sort(np.load(dossier / "ref_score_ae.npy"))
        self.ref_lstm = np.sort(np.load(dossier / "ref_score_lstm.npy"))

        if verbeux:
            print(f"Détecteur chargé : {len(self.aes)} AE + {len(self.lstms)} LSTM "
                  f"| seuil {self.seuil:.4f}")

    def _rang(self, valeurs, reference_triee):
        """Rang dans la référence (déjà triée) : fraction de réf <= valeur.
        On ne recalcule JAMAIS un rang sur le lot courant (ce serait transductif)."""
        return np.searchsorted(reference_triee, valeurs, side="right") / len(reference_triee)

    def _err_ae(self, Xn):
        return np.mean([np.mean((Xn - m.predict(Xn, verbose=0)) ** 2, axis=1)
                        for m in self.aes], axis=0)

    def _err_lstm(self, Xseq_n):
        return np.mean([np.mean((Xseq_n - m.predict(Xseq_n, verbose=0)) ** 2, axis=(1, 2))
                        for m in self.lstms], axis=0)

    def predire_lot(self, df):
        """
        df : DataFrame indexé par temps, colonnes = au moins self.features
             (+ 'Fault' optionnel).
        Retourne un DataFrame : score_fusion, anomalie (0/1), + Fault si présent.
        Les points sans séquence LSTM complète (début de journée) sont ignorés.
        """
        _ok = df[self.features].notna().all(axis=1)
        d = df[_ok].copy()

        # --- AE : erreur par point ---
        Xae = self.sc_ae.transform(d[self.features])
        s_ae = pd.Series(self._err_ae(Xae), index=d.index)

        # --- LSTM : séquences, sans jamais enjamber une nuit ---
        Xs, idx = [], []
        seg = (d.index.to_series().diff() > pd.Timedelta(f"{self.gap_min}min")).cumsum()
        for _, b in d.groupby(seg):
            v = b[self.features].values
            n = len(b)
            if n < self.seq_len:
                continue
            for st in range(n - self.seq_len + 1):
                Xs.append(v[st:st + self.seq_len])
                idx.append(b.index[st + self.seq_len - 1])
        if not Xs:
            raise ValueError("Aucune séquence LSTM possible (données trop courtes).")

        Xseq = np.array(Xs)
        Xseq_n = self.sc_lstm.transform(Xseq.reshape(-1, self._nf)).reshape(Xseq.shape)
        s_lstm = pd.Series(self._err_lstm(Xseq_n), index=pd.DatetimeIndex(idx))

        # --- fusion par rangs (contre la référence figée) ---
        idx_com = s_ae.index.intersection(s_lstm.index)
        r_ae = self._rang(s_ae.loc[idx_com].values, self.ref_ae)
        r_lstm = self._rang(s_lstm.loc[idx_com].values, self.ref_lstm)

        out = pd.DataFrame({"score_fusion": (r_ae + r_lstm) / 2}, index=idx_com)
        out["anomalie"] = (out["score_fusion"] >= self.seuil).astype(int)
        if "Fault" in df.columns:
            out["Fault"] = (df.loc[idx_com, "Fault"] != 0).astype(int)
        return out.sort_index()
