"""
Tests du traitement des messages du consumer (src/consumer.py).

Ces tests n'ont pas besoin de Kafka ni de PostgreSQL : ils appellent directement
traiter_message() avec le détecteur déployé (model_final/) et le jeu de test rejoué
(data/replay_test.csv). Ils vérifient que le consumer, une fois débarrassé de l'I/O,
produit les mêmes scores que le calcul batch, et qu'aucun message invalide ne fait
planter le traitement.

Lancer depuis la racine du dépôt :
    python -m pytest tests/ -v
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _stub_kafka_and_db():
    """kafka-python et db.py ne sont pas requis pour ces tests : on les remplace
    seulement s'ils ne peuvent pas être importés."""
    try:
        import kafka  # noqa: F401
        import kafka.structs  # noqa: F401
    except ImportError:
        kafka = types.ModuleType("kafka")
        kafka.KafkaConsumer = object
        kafka.KafkaProducer = object
        structs = types.ModuleType("kafka.structs")
        structs.OffsetAndMetadata = lambda *a: a
        structs.TopicPartition = lambda *a: a
        sys.modules.update({"kafka": kafka, "kafka.structs": structs})
    try:
        import db  # noqa: F401
    except ImportError:
        db = types.ModuleType("db")
        db.insert_measurement = lambda *a, **k: None
        db.insert_alarm = lambda *a, **k: None
        db.close_connection = lambda: None
        sys.modules["db"] = db


_stub_kafka_and_db()
_spec = importlib.util.spec_from_file_location("consumer", REPO / "src" / "consumer.py")
consumer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(consumer)

from detecteur import DetecteurPV  # noqa: E402


@pytest.fixture(scope="module")
def detecteur():
    return DetecteurPV(dossier=REPO / "model_final", verbeux=False)


@pytest.fixture(scope="module")
def batch():
    p = pd.read_csv(REPO / "model_final" / "predictions_test.csv", index_col=0, parse_dates=True)
    p.index = pd.to_datetime(p.index, utc=True)
    return p


@pytest.fixture(scope="module")
def replay(batch):
    """Fenêtre de replay autour du premier épisode de test (environ 40 messages avant,
    200 au total). Le replay complet prendrait plusieurs dizaines de minutes : un message
    déclenche une inférence complète, donc on teste une tranche qui couvre un épisode."""
    rp = pd.read_csv(REPO / "data" / "replay_test.csv")
    rp["time"] = pd.to_datetime(rp["time"], utc=True)
    premier_defaut = batch.index[batch["Fault"] > 0][0]
    debut = int((rp["time"] >= premier_defaut).values.argmax()) - 40
    return rp.iloc[max(debut, 0):max(debut, 0) + 200].reset_index(drop=True)


def _rejouer(rows, detecteur, offset_depart=0):
    """Fait passer des lignes dans traiter_message, comme le ferait le consumer."""
    tampon = consumer.TamponGlissant(detecteur.seq_len, detecteur.gap_min)
    resultats = []
    for i, row in enumerate(rows):
        resultats.append(consumer.traiter_message(dict(row), offset_depart + i, tampon, detecteur))
    return resultats, tampon


def test_seuil_filtre_diurne():
    assert consumer.FILTRE_GTI_MIN == 50


def test_replay_reproduit_le_batch(detecteur, replay, batch):
    resultats, _ = _rejouer(replay.to_dict("records"), detecteur)
    scores = {}
    for row, r in zip(replay.to_dict("records"), resultats):
        if r["score"] is not None:
            scores[row["time"]] = (r["score"], r["anomalie"])
    assert len(scores) > 100, "la tranche de replay doit contenir des points scorés"
    idx = pd.DatetimeIndex(list(scores.keys()))
    s_stream = np.array([v[0] for v in scores.values()])
    a_stream = np.array([v[1] for v in scores.values()])
    s_batch = batch["score_fusion"].reindex(idx).values
    a_batch = batch["anomalie"].reindex(idx).values
    assert not np.isnan(s_batch).any(), "chaque point scoré doit exister dans le batch"
    assert np.nanmax(np.abs(s_stream - s_batch)) < 1e-3
    assert (a_stream == a_batch).all(), "les alarmes doivent être identiques au batch"
    assert a_stream.sum() > 0, "la tranche doit contenir au moins une alarme (épisode de test)"


def test_premiers_points_de_fenetre_non_scores(detecteur, replay):
    resultats, _ = _rejouer(replay.to_dict("records"), detecteur)
    assert all(r["raison"] == "fenêtre incomplète" for r in resultats[:5]), \
        "les 5 premiers points n'ont pas de fenêtre complète"
    assert all(r["score"] is None for r in resultats if r["raison"] is not None)


def test_offsets_commit_non_decroissants(detecteur, replay):
    resultats, _ = _rejouer(replay.to_dict("records"), detecteur)
    commits = [r["offset_commit"] for r in resultats]
    assert all(b >= a for a, b in zip(commits, commits[1:])), "l'offset commité ne doit jamais reculer"


def test_message_sans_horodatage_ignore(detecteur):
    tampon = consumer.TamponGlissant(detecteur.seq_len, detecteur.gap_min)
    for mesure in [{"GTI": 300.0}, {"time": None, "GTI": 300.0}, {"time": "pas une date", "GTI": 300.0}]:
        r = consumer.traiter_message(mesure, 42, tampon, detecteur)
        assert r["horodatage"] is None
        assert r["score"] is None
        assert r["raison"] and "horodatage" in r["raison"]
        assert r["offset_commit"] == 43, "tampon vide : on committe offset + 1 (convention Kafka)"


def test_nuit_ignoree_sans_toucher_au_tampon(detecteur, replay):
    tampon = consumer.TamponGlissant(detecteur.seq_len, detecteur.gap_min)
    base = replay.iloc[0].to_dict()
    for gti in [0.0, 50.0, float("nan"), None]:
        mesure = {**base, "GTI": gti, "time": "2022-05-01 00:00:00+00:00"}
        r = consumer.traiter_message(mesure, 7, tampon, detecteur)
        assert r["score"] is None
        assert r["raison"] == "nuit ou irradiance absente"
        assert r["offset_commit"] == 8, "tampon vide : on committe offset + 1 (convention Kafka)"
    assert len(tampon.points) == 0


def test_irradiance_juste_au_dessus_du_seuil_est_scoree_ou_attendue(detecteur, replay):
    tampon = consumer.TamponGlissant(detecteur.seq_len, detecteur.gap_min)
    base = replay.iloc[100].to_dict()
    r = consumer.traiter_message({**base, "GTI": 50.0}, 1, tampon, detecteur)
    assert r["raison"] == "nuit ou irradiance absente"
    r = consumer.traiter_message({**base, "GTI": 50.01}, 2, tampon, detecteur)
    assert r["raison"] != "nuit ou irradiance absente"


def test_capteur_manquant_est_signale_et_non_score(detecteur, replay):
    rows = replay.iloc[100:112].to_dict("records")
    rows[6] = {**rows[6], "Va": float("nan")}
    resultats, _ = _rejouer(rows, detecteur)
    for r in resultats:
        if r["score"] is None:
            assert r["raison"], "un point non scoré doit avoir une raison"
        else:
            assert r["raison"] is None
    assert any(r["raison"] for r in resultats), "la valeur manquante doit apparaître dans les raisons"


def test_scorer_ne_masque_pas_les_erreurs(detecteur):
    class DetecteurDefaillant:
        def predire_lot(self, fenetre):
            raise ValueError("erreur simulée")

    fenetre = pd.DataFrame({"GTI": [300.0]}, index=pd.DatetimeIndex(["2022-05-01 10:00"], tz="UTC"))
    score, anomalie, raison = consumer.scorer(DetecteurDefaillant(), fenetre)
    assert score is None and anomalie is None
    assert raison and "erreur simulée" in raison


def test_tampon_se_vide_apres_une_nuit():
    tampon = consumer.TamponGlissant(seq_len=3, gap_minutes=7)
    t0 = pd.Timestamp("2022-05-01 10:00", tz="UTC")
    for k in range(3):
        tampon.ajouter(k, t0 + pd.Timedelta(minutes=5 * k), {"x": k})
    assert len(tampon.points) == 3
    assert tampon.ajouter(3, t0 + pd.Timedelta(hours=12), {"x": 9}) is None
    assert len(tampon.points) == 1


# --- Garanties de livraison (ajoutées avec la correction de la durabilité des alarmes) ---

def test_redemarrage_ne_rescore_pas_le_dernier_point(detecteur, replay):
    """Après un arrêt propre, la relecture reconstruit la fenêtre sans rescorer
    le dernier point traité, puis le point suivant obtient le même score."""
    rows = replay.to_dict("records")
    ref, _ = _rejouer(rows, detecteur)
    k = next(i for i, r in enumerate(ref) if r["score"] is not None and i > 20)
    reprise = ref[k]["offset_commit"]
    assert reprise <= k, "la reprise doit relire une partie de la fenêtre"
    res, _ = _rejouer(rows[reprise:k + 2], detecteur, offset_depart=reprise)
    assert res[k - reprise]["score"] is None, "le dernier point traité ne doit pas être rescoré"
    assert res[k + 1 - reprise]["score"] == pytest.approx(ref[k + 1]["score"], abs=1e-9)


class _Future:
    def __init__(self, exc=None): self.exc = exc
    def get(self, timeout=None):
        if self.exc: raise self.exc
        return "ok"


class _Producer:
    def __init__(self, exc=None): self.exc, self.envois = exc, []
    def send(self, topic, key=None, value=None):
        self.envois.append((topic, key, value)); return _Future(self.exc)


class _Consumer:
    def __init__(self): self.commits = []
    def commit(self, offsets): self.commits.append(offsets)


class _Message:
    def __init__(self, offset, value): self.topic, self.partition, self.offset, self.value = "measurements", 0, offset, value


def test_publier_alarme_attend_l_accuse_et_cle_horodatage():
    p = _Producer()
    consumer.publier_alarme(p, {"time": "2022-05-01T10:00:00+00:00", "GTI": 300.0, "Ig": float("nan")}, 0.97)
    topic, cle, valeur = p.envois[0]
    assert topic == "alarms" and cle == b"2022-05-01T10:00:00+00:00"
    assert valeur["Ig"] is None and valeur["score_fusion"] == 0.97
    with pytest.raises(TimeoutError):
        consumer.publier_alarme(_Producer(TimeoutError("pas d'accusé")), {"time": "t"}, 0.9)


def test_pas_de_commit_si_la_publication_echoue(detecteur, replay, batch, monkeypatch):
    """Si Kafka ne confirme pas l'alarme, l'offset n'est PAS committé."""
    monkeypatch.setattr(consumer, "insert_measurement", lambda *a, **k: None)
    monkeypatch.setattr(consumer, "insert_alarm", lambda *a, **k: None)
    rows = replay.to_dict("records")
    alarmes = set(batch.index[batch["anomalie"] == 1])
    tampon = consumer.TamponGlissant(detecteur.seq_len, detecteur.gap_min)
    c, stats = _Consumer(), {"lues": 0, "scorees": 0, "alarmes": 0, "raisons": {}}
    for i, row in enumerate(rows):
        msg = _Message(i, dict(row))
        if pd.Timestamp(row["time"]) in alarmes:
            avant = len(c.commits)
            with pytest.raises(ConnectionError):
                consumer.traiter_et_persister(msg, tampon, detecteur, c,
                                              _Producer(ConnectionError("broker perdu")), stats)
            assert len(c.commits) == avant, "aucun commit après un échec de publication"
            return
        consumer.traiter_et_persister(msg, tampon, detecteur, c, _Producer(), stats)
    pytest.fail("la tranche devait contenir une alarme")


def test_sans_nan_produit_du_json_standard():
    import json
    propre = consumer.sans_nan({"a": float("nan"), "b": float("inf"), "c": 1.5, "d": "x"})
    assert propre == {"a": None, "b": None, "c": 1.5, "d": "x"}
    json.dumps(propre, allow_nan=False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
