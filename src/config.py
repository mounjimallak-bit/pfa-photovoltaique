import os
from pathlib import Path
from dotenv import load_dotenv

# Le même .env est lu par docker compose : une seule source de valeurs.
# override=False : une variable déjà définie dans le shell reste prioritaire.
FICHIER_ENV = Path(os.getenv("PFA_ENV_FILE") or Path(__file__).resolve().parents[1] / ".env")
if FICHIER_ENV.is_file():
    load_dotenv(FICHIER_ENV, override=False)


class ErreurConfiguration(ValueError):
    """Variable d'environnement présente mais invalide."""


def _texte(nom, defaut):
    return os.getenv(nom, "").strip() or defaut


def _nombre(nom, defaut, minimum, type_=int):
    brut = os.getenv(nom, "").strip()
    if not brut:
        return defaut
    try:
        valeur = type_(brut)
    except ValueError:
        raise ErreurConfiguration(
            f"{nom}={brut!r} : nombre attendu ({type_.__name__})") from None
    if valeur < minimum:
        raise ErreurConfiguration(f"{nom}={valeur} : minimum {minimum}")
    return valeur


def _brokers(nom, defaut):
    valeur = _texte(nom, defaut)
    for adresse in valeur.split(","):
        hote, deux_points, port = adresse.strip().rpartition(":")
        if not (deux_points and hote and port.isdigit()):
            raise ErreurConfiguration(
                f"{nom}={valeur!r} : format attendu hote:port "
                f"(plusieurs : séparés par des virgules)")
    return valeur


# --- Kafka ---
KAFKA_BROKER = _brokers("KAFKA_BROKER", "localhost:9092")
TOPIC_MESURES = _texte("KAFKA_TOPIC_MEASUREMENTS", "measurements")
TOPIC_ALARMES = _texte("KAFKA_TOPIC_ALARMS", "alarms")
GROUP_ID = _texte("KAFKA_GROUP_ID", "pfa-detecteur")
REQUEST_TIMEOUT_MS = _nombre("KAFKA_REQUEST_TIMEOUT_MS", 10000, minimum=1000)
DELAI_LIVRAISON_MS = _nombre("ALARM_DELIVERY_TIMEOUT_MS", 30000, minimum=1000)

# --- Replay ---
DELAI_REPLAY_S = _nombre("REPLAY_DELAY_S", 1.0, minimum=0.0, type_=float)

# --- PostgreSQL / TimescaleDB (noms standard de libpq) ---
PG_HOST = _texte("PGHOST", "localhost")
PG_PORT = _nombre("PGPORT", 5432, minimum=1)
PG_DATABASE = _texte("PGDATABASE", "photovoltaique")
PG_USER = _texte("PGUSER", "pfa")
PG_PASSWORD = os.getenv("PGPASSWORD", "").strip() or None
PG_CONNECT_TIMEOUT = _nombre("PGCONNECT_TIMEOUT", 5, minimum=1)

# --- Cohérence entre valeurs ---
if DELAI_LIVRAISON_MS < REQUEST_TIMEOUT_MS:
    raise ErreurConfiguration(
        f"ALARM_DELIVERY_TIMEOUT_MS ({DELAI_LIVRAISON_MS}) doit être au moins égal à "
        f"KAFKA_REQUEST_TIMEOUT_MS ({REQUEST_TIMEOUT_MS}) : règle imposée par kafka-python")
if TOPIC_MESURES == TOPIC_ALARMES:
    raise ErreurConfiguration(
        f"KAFKA_TOPIC_MEASUREMENTS et KAFKA_TOPIC_ALARMS sont identiques ({TOPIC_MESURES!r})")


def resume():
    """Une ligne à afficher au démarrage de chaque composant."""
    return (f"Kafka {KAFKA_BROKER} | topics {TOPIC_MESURES} -> {TOPIC_ALARMES} "
            f"| groupe {GROUP_ID} | base {PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}")





def exiger_secrets():
    """À appeler par tout composant qui se connecte à la base."""
    manquants = [nom for nom, valeur in (("PGPASSWORD", PG_PASSWORD),) if not valeur]
    if manquants:
        raise ErreurConfiguration(
            f"Variable obligatoire manquante : {', '.join(manquants)}. "
            f"Copier .env.example en .env et y définir la valeur (voir README).")
