"""
Replayer : rejoue la partition de TEST ligne par ligne sur Kafka.
Simule un flux temps réel pour la démo.
"""
from pathlib import Path
import json
import time

import pandas as pd
from kafka import KafkaProducer


RACINE = Path(__file__).resolve().parents[1]

KAFKA_BROKER = "localhost:9092"
TOPIC = "measurements"

# Extrait versionné de la partition de test (~2 975 points), rejouable depuis un
# clone nu. Il est dérivé de data/df_etude_split.csv, produit par le § 5.3 du
# notebook, qui lui n'est pas versionné : si le cache manque, on le reconstruit.
CACHE = RACINE / "data" / "replay_test.csv"
SOURCE = RACINE / "data" / "df_etude_split.csv"
PARTITION = "test"


def construire_cache() -> pd.DataFrame:
    """Extrait la partition de test de df_etude_split.csv et l'écrit dans CACHE."""
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"{CACHE.name} et {SOURCE.name} sont tous deux absents. Exécuter "
            f"notebooks/final.ipynb (§ 5.3) pour régénérer {SOURCE.name}.")
    df = pd.read_csv(SOURCE)
    df = df[df["split"] == PARTITION].drop(columns=["split", "fgroup"], errors="ignore")
    df.to_csv(CACHE, index=False)
    print(f"{CACHE.name} reconstruit depuis {SOURCE.name} : {len(df)} points")
    return df


def charger_partition() -> pd.DataFrame:
    """Partition de test, triée chronologiquement."""
    df = pd.read_csv(CACHE) if CACHE.exists() else construire_cache()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time")


def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def replay(delay: float = 1.0):
    """
    Publie chaque mesure de la partition de test sur Kafka.
    delay = secondes entre deux messages (simuler le temps réel).

    Le consumer met ~0,9 s à scorer un point (10 réseaux interrogés un par un) :
    en deçà de 1 s le flux prend du retard. Ça reste 300x plus rapide que la
    cadence réelle des mesures, qui est de 5 minutes.
    """
    producer = create_producer()
    df = charger_partition()

    print(f"Replayer : {len(df)} mesures ({PARTITION}) à envoyer sur '{TOPIC}'")

    for _, row in df.iterrows():
        message = row.to_dict()
        message["time"] = row["time"].isoformat()   # JSON ne sérialise pas un Timestamp
        producer.send(TOPIC, value=message)
        time.sleep(delay)

    producer.flush()
    print("Replayer terminé.")


if __name__ == "__main__":
    replay()
