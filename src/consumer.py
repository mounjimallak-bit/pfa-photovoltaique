"""
Consumer Kafka : lit les mesures, score avec le système de détection,
écrit dans TimescaleDB, publie les alarmes.
"""
from collections import deque
import json

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

from db import insert_measurement, insert_alarm
from detecteur import DetecteurPV


KAFKA_BROKER = "localhost:9092"
TOPIC_IN = "measurements"
TOPIC_ALARMS = "alarms"


class TamponGlissant:
    """
    Accumule les dernières mesures pour reconstituer une séquence LSTM.

    Le détecteur travaille par lot et exige seq_len points contigus, alors que
    Kafka livre point par point. Le tampon est remis à zéro dès qu'un trou
    dépasse gap_minutes : sans ça, on fabriquerait des transitions soir -> matin
    qui n'existent pas dans les données, la nuit n'étant pas mesurée.
    """

    def __init__(self, seq_len: int, gap_minutes: int):
        self.seq_len = seq_len
        self.gap = pd.Timedelta(f"{gap_minutes}min")
        self.points = deque(maxlen=seq_len)

    def ajouter(self, horodatage, mesure: dict):
        """Retourne un DataFrame de seq_len lignes si la fenêtre est complète."""
        if self.points and horodatage - self.points[-1][0] > self.gap:
            self.points.clear()
        self.points.append((horodatage, mesure))

        if len(self.points) < self.seq_len:
            return None
        return pd.DataFrame([m for _, m in self.points],
                            index=pd.DatetimeIndex([t for t, _ in self.points]))


def run_consumer():
    detecteur = DetecteurPV()
    tampon = TamponGlissant(detecteur.seq_len, detecteur.gap_min)

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
    n_lues = n_scorees = n_alarmes = 0

    for message in consumer:
        mesure = message.value
        n_lues += 1

        horodatage = pd.Timestamp(mesure["time"])
        fenetre = tampon.ajouter(horodatage, {c: mesure.get(c) for c in detecteur.features})
        if fenetre is None:
            continue   # fenêtre incomplète : début de série ou reprise après la nuit

        resultat = detecteur.predire_lot(fenetre)
        if resultat.empty:
            continue   # valeur capteur manquante sur la fenêtre

        score = float(resultat["score_fusion"].iloc[-1])
        est_anomalie = bool(resultat["anomalie"].iloc[-1])
        n_scorees += 1

        # La table measurements utilise des colonnes en minuscules, le flux
        # transporte les noms de capteurs d'origine (GTI, Pg, Va...).
        insert_measurement({k.lower(): v for k, v in mesure.items()}, score, est_anomalie)

        if est_anomalie:
            n_alarmes += 1
            producer.send(TOPIC_ALARMS, value={**mesure, "score_fusion": score})
            insert_alarm(mesure, score)
            print(f"ALARME {horodatage} | score {score:.4f} "
                  f"(seuil {detecteur.seuil:.4f})")

        if n_scorees % 100 == 0:
            print(f"{n_lues} mesures lues | {n_scorees} scorées | {n_alarmes} alarmes")


if __name__ == "__main__":
    run_consumer()
