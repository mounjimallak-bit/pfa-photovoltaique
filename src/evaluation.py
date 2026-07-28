"""
Évaluation partagée des détecteurs d'anomalies.

Tout modèle (IF, Autoencoder, LSTM-AE, fusion) passe par ces fonctions.
Convention de labellisation séquentielle : une séquence porte le label de son
DERNIER point (sémantique temps réel : « à cet instant, alarme ou pas »).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score)


def metriques(y_true, y_pred, scores=None, nom="modele"):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    r = {"Modele": nom, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
         "Precision": prec, "Recall": rec, "F1": f1,
         "Taux_alerte": (tp + fp) / len(y_true),
         "Prevalence": float(y_true.mean()), "N": len(y_true)}

    if scores is not None:
        s = np.asarray(scores, dtype=float)
        r["ROC_AUC"] = roc_auc_score(y_true, s)
        r["PR_AUC"]  = average_precision_score(y_true, s)
        # La PR-AUC dépend du taux de base : 0.42 à 3.6 % d'anomalies vaut MIEUX
        # que 0.87 à 18.6 %. Le lift (PR-AUC / prévalence) rend val et test comparables.
        r["PR_AUC_lift"] = r["PR_AUC"] / r["Prevalence"] if r["Prevalence"] else np.nan
    return r


def recall_par_type(f_true, y_pred, nom="modele"):
    """Recall détaillé par type de panne — c'est là que se départagent les modèles."""
    f_true = np.asarray(f_true, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)
    lignes = [{"Fault": t, "Occurrences": int((f_true == t).sum()),
               f"Recall {nom}": float(y_pred[f_true == t].mean())}
              for t in sorted(set(f_true[f_true != 0]))]
    return pd.DataFrame(lignes)


def seuil_pour_budget(scores, n_alertes):
    """Seuil déclenchant exactement n_alertes — pour comparer à budget d'alarme égal."""
    s = np.asarray(scores, dtype=float)
    n = min(max(int(n_alertes), 1), len(s))
    return float(np.partition(s, -n)[-n])


def predire_a_budget(scores, n_alertes):
    return (np.asarray(scores, dtype=float) >= seuil_pour_budget(scores, n_alertes)).astype(int)


def masque_commun(index_ref, index_modele):
    """
    Masque booléen sur index_ref, restreint aux instants qu'un modèle séquentiel
    peut réellement scorer. Indispensable pour comparer IF (ponctuel, 5 056 points)
    et LSTM-AE (séquentiel, 4 504 points) sur la MÊME base.
    """
    return pd.Index(index_ref).isin(pd.Index(index_modele))


def comparer(resultats):
    df = pd.DataFrame(resultats).set_index("Modele")
    ordre = ["N", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1",
             "ROC_AUC", "PR_AUC", "PR_AUC_lift", "Taux_alerte", "Prevalence"]
    return df[[c for c in ordre if c in df.columns]].round(4)

def episodes(y, index, segments=None):
    """Blocs contigus d'anomalie. `segments` = liste (a,b) pour couper aux nuits."""
    y = np.asarray(y).astype(int)
    blocs = segments if segments is not None else [(0, len(y))]
    eps = []
    for a, b in blocs:
        d = np.flatnonzero(np.diff(np.r_[0, y[a:b], 0]) != 0)
        eps += [(a + s, a + e) for s, e in zip(d[::2], d[1::2])]
    return eps


def metriques_episode(y, y_pred, eps, pas_min=5, nom="modele"):
    """Détection au niveau incident : ce que voit un exploitant."""
    y_pred = np.asarray(y_pred).astype(int)
    lat = [np.flatnonzero(y_pred[a:b])[0] for a, b in eps if y_pred[a:b].any()]
    fp_pts = int(y_pred[np.asarray(y) == 0].sum())
    return {"Modele": nom, "Episodes": len(eps), "Detectes": len(lat),
            "Recall_episode": len(lat) / len(eps) if eps else np.nan,
            "Latence_med_min": pas_min * float(np.median(lat)) if lat else np.nan,
            "FP_points": fp_pts,
            "FP_par_jour": fp_pts / max(len(set(np.arange(len(y_pred)))) , 1)}