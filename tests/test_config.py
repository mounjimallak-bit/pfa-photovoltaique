
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import config  # noqa: E402

VARIABLES = ("KAFKA_BROKER", "KAFKA_TOPIC_MEASUREMENTS", "KAFKA_TOPIC_ALARMS",
             "KAFKA_GROUP_ID", "KAFKA_REQUEST_TIMEOUT_MS", "ALARM_DELIVERY_TIMEOUT_MS",
             "REPLAY_DELAY_S", "PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD",
             "PGCONNECT_TIMEOUT")


@pytest.fixture
def charger(monkeypatch):
    """Recharge config avec exactement les variables données, sans lire de .env."""
    def _charger(**env):
        for nom in VARIABLES:
            monkeypatch.delenv(nom, raising=False)
        monkeypatch.setenv("PFA_ENV_FILE", str(REPO / "absent.env"))
        for nom, valeur in env.items():
            monkeypatch.setenv(nom, valeur)
        return importlib.reload(config)
    yield _charger
    monkeypatch.undo()
    importlib.reload(config)


def test_valeurs_par_defaut(charger):
    c = charger()
    assert c.KAFKA_BROKER == "localhost:9092"
    assert (c.TOPIC_MESURES, c.TOPIC_ALARMES) == ("measurements", "alarms")
    assert c.GROUP_ID == "pfa-detecteur"
    assert (c.REQUEST_TIMEOUT_MS, c.DELAI_LIVRAISON_MS, c.DELAI_REPLAY_S) == (10000, 30000, 1.0)
    assert (c.PG_HOST, c.PG_PORT, c.PG_DATABASE, c.PG_USER) == (
        "localhost", 5432, "photovoltaique", "pfa")


def test_valeurs_personnalisees(charger):
    c = charger(KAFKA_BROKER="kafka:29092", REPLAY_DELAY_S="0.5", PGPORT="5433")
    assert (c.KAFKA_BROKER, c.DELAI_REPLAY_S, c.PG_PORT) == ("kafka:29092", 0.5, 5433)


def test_variable_vide_prend_la_valeur_par_defaut(charger):
    c = charger(KAFKA_BROKER="", KAFKA_GROUP_ID="  ", PGPORT="")
    assert (c.KAFKA_BROKER, c.GROUP_ID, c.PG_PORT) == ("localhost:9092", "pfa-detecteur", 5432)


# ValueError et non config.ErreurConfiguration : reload() recrée la classe à chaque
# chargement, l'ancienne ne correspondrait plus à l'exception levée.
@pytest.mark.parametrize("env, morceau", [
    ({"ALARM_DELIVERY_TIMEOUT_MS": "30s"}, "ALARM_DELIVERY_TIMEOUT_MS"),
    ({"PGPORT": "-1"}, "PGPORT"),
    ({"KAFKA_BROKER": "localhost;9092"}, "KAFKA_BROKER"),
    ({"KAFKA_BROKER": "localhost"}, "KAFKA_BROKER"),
    ({"ALARM_DELIVERY_TIMEOUT_MS": "5000"}, "KAFKA_REQUEST_TIMEOUT_MS"),
    ({"KAFKA_TOPIC_ALARMS": "measurements"}, "identiques"),
    ({"REPLAY_DELAY_S": "-1"}, "REPLAY_DELAY_S"),
])
def test_configuration_invalide_refusee(charger, env, morceau):
    with pytest.raises(ValueError, match=morceau):
        charger(**env)


def test_aucune_configuration_en_dur_dans_les_composants():
    for nom in ("consumer.py", "replayer.py", "db.py"):
        texte = (REPO / "src" / nom).read_text(encoding="utf-8")
        assert "os.getenv" not in texte, f"{nom} lit une variable sans passer par config.py"
        assert "localhost:9092" not in texte, f"{nom} contient une adresse Kafka en dur"


def test_mot_de_passe_sans_valeur_par_defaut(charger):
    c = charger()
    assert c.PG_PASSWORD is None
    with pytest.raises(ValueError, match="PGPASSWORD"):
        c.exiger_secrets()


def test_mot_de_passe_defini_et_jamais_affiche(charger):
    c = charger(PGPASSWORD="valeur-de-test")
    c.exiger_secrets()
    assert "valeur-de-test" not in c.resume()
