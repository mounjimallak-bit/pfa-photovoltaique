from collections import deque
import json
import os
import signal
import sys

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import OffsetAndMetadata, TopicPartition

from db import insert_measurement, insert_alarm, close_connection
from detecteur import DetecteurPV


KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_IN = "measurements"
TOPIC_ALARMS = "alarms"
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "pfa-detecteur")
# Même seuil que le notebook : seules les mesures diurnes (GTI > 50 W/m²) sont scorées.
# Sans ce filtre, un flux continu enverrait des données nocturnes aux modèles.
FILTRE_GTI_MIN = 50


class TamponGlissant:
    """Accumule les mesures récentes pour reconstruire une séquence LSTM.

    Le tampon est vidé dès qu'un écart dépasse gap_minutes : sinon on fabriquerait
    des transitions soir -> matin qui n'existent pas, la nuit n'étant pas mesurée.
    Chaque point garde son offset Kafka, pour que le commit pointe sur le plus ancien
    point du tampon et qu'un redémarrage reconstruise la fenêtre.
    """

    def __init__(self, seq_len: int, gap_minutes: int):
        self.seq_len = seq_len
        self.gap = pd.Timedelta(f"{gap_minutes}min")
        self.points = deque(maxlen=seq_len)

    def ajouter(self, offset: int, horodatage, mesure: dict):
        """Retourne un DataFrame de seq_len lignes si la fenêtre est complète."""
        if self.points and horodatage - self.points[-1][1] > self.gap:
            self.points.clear()
        self.points.append((offset, horodatage, mesure))

        if len(self.points) < self.seq_len:
            return None
        return pd.DataFrame(
            [m for _, _, m in self.points],
            index=pd.DatetimeIndex([t for _, t, _ in self.points]))

    def offset_a_commiter(self):
        return self.points[0][0]


def scorer(detecteur, fenetre):
    """Retourne (score, anomalie, raison_non_score). raison_non_score est None si scoré."""
    if fenetre is None:
        return None, None, "fenêtre incomplète"
    try:
        resultat = detecteur.predire_lot(fenetre)
    except ValueError as e:
        # Ne plus avaler l'erreur en silence : elle est remontée au journal.
        return None, None, f"scoring impossible ({e})"
    if resultat.empty:
        return None, None, "aucune séquence complète"
    return (float(resultat["score_fusion"].iloc[-1]),
            bool(resultat["anomalie"].iloc[-1]), None)


def traiter_message(mesure, offset, tampon, detecteur):
    """Traite un message sans I/O Kafka ni base, pour pouvoir le tester.

    Retourne un dict : horodatage (None si invalide), score, anomalie,
    raison (None si scoré), offset_commit (offset à commiter une fois le message traité).
    """
    def offset_a_commiter():
        return tampon.offset_a_commiter() if tampon.points else offset

    try:
        horodatage = pd.Timestamp(mesure["time"])
        if pd.isna(horodatage):
            raise ValueError("horodatage vide")
    except (KeyError, TypeError, ValueError) as e:
        # Sans horodatage, la mesure ne peut être ni rangée ni scorée : on la signale
        # et on avance l'offset, sinon le message serait relu et ferait planter le
        # consumer à chaque redémarrage.
        return dict(horodatage=None, score=None, anomalie=None,
                    raison=f"horodatage invalide ({e})", offset_commit=offset_a_commiter())

    gti = mesure.get("GTI")
    if gti is None or pd.isna(gti) or not gti > FILTRE_GTI_MIN:
        return dict(horodatage=horodatage, score=None, anomalie=None,
                    raison="nuit ou irradiance absente", offset_commit=offset_a_commiter())

    fenetre = tampon.ajouter(offset, horodatage, {c: mesure.get(c) for c in detecteur.features})
    score, est_anomalie, raison = scorer(detecteur, fenetre)
    return dict(horodatage=horodatage, score=score, anomalie=est_anomalie,
                raison=raison, offset_commit=offset_a_commiter())


def run_consumer():
    detecteur = DetecteurPV()
    tampon = TamponGlissant(detecteur.seq_len, detecteur.gap_min)

    consumer = KafkaConsumer(
        TOPIC_IN,
        bootstrap_servers=KAFKA_BROKER,
        group_id=GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
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

    raisons = {}
    for message in consumer:
        mesure = message.value
        n_lues += 1

        r = traiter_message(mesure, message.offset, tampon, detecteur)
        if r["score"] is not None:
            n_scorees += 1
        if r["raison"] is not None:
            raisons[r["raison"]] = raisons.get(r["raison"], 0) + 1

        if r["horodatage"] is not None:
            insert_measurement({k.lower(): v for k, v in mesure.items()},
                               r["score"], r["anomalie"])

        if r["anomalie"]:
            n_alarmes += 1
            producer.send(TOPIC_ALARMS, value={**mesure, "score_fusion": r["score"]})
            insert_alarm(mesure, r["score"])
            print(f"ALARME {r['horodatage']} | score {r['score']:.4f} "
                  f"(seuil {detecteur.seuil:.4f})")

        # Commit en dernier, une fois la mesure ET l'alarme durables : commiter
        # avant l'envoi de l'alarme la perdrait définitivement en cas de crash
        # entre les deux, le message n'étant plus redélivré.
        consumer.commit({
            TopicPartition(message.topic, message.partition):
                OffsetAndMetadata(r["offset_commit"], None),
        })

        if n_lues % 100 == 0:
            print(f"{n_lues} mesures lues | {n_scorees} scorées "
                  f"| {n_lues - n_scorees} non scorables | {n_alarmes} alarmes | {raisons}")


if __name__ == "__main__":
    run_consumer()
