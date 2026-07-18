"""
Replayer : rejoue le dataset TEST ligne par ligne sur Kafka.
Simule un flux temps réel pour la démo.
"""
import json
import time
import pandas as pd
from kafka import KafkaProducer


KAFKA_BROKER = "localhost:9092"
TOPIC = "measurements"


def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def replay(csv_path: str, delay: float = 0.5):
    """
    Lit le CSV de test et publie chaque ligne sur Kafka.
    delay = secondes entre chaque message (simuler le temps réel).
    """
    producer = create_producer()
    df = pd.read_csv(csv_path)

    print(f"Replayer : {len(df)} mesures à envoyer sur '{TOPIC}'")

    for _, row in df.iterrows():
        message = row.to_dict()
        producer.send(TOPIC, value=message)
        time.sleep(delay)

    producer.flush()
    print("Replayer terminé.")


if __name__ == "__main__":
    # TODO S3 : remplacer par le chemin du fichier test
    replay("data/test_data.csv", delay=0.5)
