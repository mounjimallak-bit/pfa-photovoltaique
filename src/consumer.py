"""
Consumer Kafka : lit les mesures, score avec le système de détection,
écrit dans TimescaleDB, publie les alarmes.
"""
from collections import deque
import json
import os
import signal
import sys

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

from db import insert_measurement, insert_alarm, close_connection
from detecteur import DetecteurPV


KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_IN = "measurements"
TOPIC_ALARMS = "alarms"

# Un group_id est indispensable : sans lui, kafka-python ne commite aucun
# offset et chaque redémarrage relit le topic depuis le début. Combiné à
# auto_offset_reset="earliest", rejouer la démo réinsérait deux fois les mêmes
# 2 813 points. Le group_id fait reprendre le consumer là où il s'était arrêté ;
# les insertions restent idempotentes par sécurité (ON CONFLICT, cf. db.py).
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "pfa-detecteur")


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


def scorer(detecteur, fenetre):
    """
    Score le dernier point d'une fenêtre complète.

    Retourne (score, anomalie) ou (None, False) si la fenêtre n'est pas
    scorable. predire_lot lève une ValueError dès qu'une seule des seq_len
    lignes porte un NaN sur une feature : elle est écartée par le filtre
    notna(), le segment retombe sous seq_len et plus aucune séquence n'est
    constructible. Un capteur qui décroche une fois suffisait à interrompre la
    boucle Kafka — d'où le rattrapage ici.
    """
    if fenetre is None:
        return None, False
    try:
        resultat = detecteur.predire_lot(fenetre)
    except ValueError:
        return None, False
    if resultat.empty:
        return None, False
    return float(resultat["score_fusion"].iloc[-1]), bool(resultat["anomalie"].iloc[-1])


def run_consumer():
    detecteur = DetecteurPV()
    tampon = TamponGlissant(detecteur.seq_len, detecteur.gap_min)

    consumer = KafkaConsumer(
        TOPIC_IN,
        bootstrap_servers=KAFKA_BROKER,
        group_id=GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    def arreter(*_):
        print("\nArrêt demandé — fermeture propre.")
        producer.flush()
        consumer.close()
        close_connection()
        sys.exit(0)

    signal.signal(signal.SIGINT, arreter)
    signal.signal(signal.SIGTERM, arreter)

    print(f"Consumer en écoute sur '{TOPIC_IN}' (groupe '{GROUP_ID}')...")
    n_lues = n_scorees = n_alarmes = 0

    for message in consumer:
        mesure = message.value
        n_lues += 1

        horodatage = pd.Timestamp(mesure["time"])
        fenetre = tampon.ajouter(horodatage,
                                 {c: mesure.get(c) for c in detecteur.features})
        score, est_anomalie = scorer(detecteur, fenetre)
        if score is not None:
            n_scorees += 1

        # La mesure est écrite dans TOUS les cas, scorable ou non : les
        # seq_len - 1 premiers points de chaque journée n'ont pas de séquence
        # complète (162 points sur 2 975 pour la partition de test). Les
        # exclure laissait autant de trous dans Grafana sur des relevés qui
        # existent bel et bien. anomaly_score reste NULL pour ces points.
        # La table utilise des colonnes en minuscules, le flux transporte les
        # noms de capteurs d'origine (GTI, Pg, Va...).
        insert_measurement({k.lower(): v for k, v in mesure.items()},
                           score, est_anomalie)

        if est_anomalie:
            n_alarmes += 1
            producer.send(TOPIC_ALARMS, value={**mesure, "score_fusion": score})
            insert_alarm(mesure, score)
            print(f"ALARME {horodatage} | score {score:.4f} "
                  f"(seuil {detecteur.seuil:.4f})")

        if n_lues % 100 == 0:
            print(f"{n_lues} mesures lues | {n_scorees} scorées "
                  f"| {n_lues - n_scorees} non scorables | {n_alarmes} alarmes")


if __name__ == "__main__":
    run_consumer()
