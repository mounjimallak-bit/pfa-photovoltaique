"""
Consumer Kafka : lit les mesures, score avec le modèle ML,
écrit dans TimescaleDB, publie les alarmes.
"""
import json
import joblib
from kafka import KafkaConsumer, KafkaProducer
from db import insert_measurement, insert_alarm


KAFKA_BROKER = "localhost:9092"
TOPIC_IN = "measurements"
TOPIC_ALARMS = "alarms"


def load_model(model_path: str = "models/isolation_forest.pkl"):
    """Charge le modèle entraîné UNE FOIS au démarrage."""
    print(f"Chargement du modèle : {model_path}")
    return joblib.load(model_path)


def run_consumer():
    model = load_model()

    consumer = KafkaConsumer(
        TOPIC_IN,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Consumer en écoute sur '{TOPIC_IN}'...")

    for message in consumer:
        mesure = message.value
        # TODO S4 : extraire les features, predict
        # score = model.predict([features])
        # is_anomaly = (score == -1)

        # insert_measurement(mesure, anomaly_score, is_anomaly)

        # if is_anomaly:
        #     producer.send(TOPIC_ALARMS, value=mesure)
        #     insert_alarm(mesure, anomaly_score)
        pass


if __name__ == "__main__":
    run_consumer()
