from collections import deque
import json
import math
import signal

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import OffsetAndMetadata, TopicPartition

from config import (
    KAFKA_BROKER, GROUP_ID, REQUEST_TIMEOUT_MS, DELAI_LIVRAISON_MS,
    TOPIC_MESURES as TOPIC_IN, TOPIC_ALARMES as TOPIC_ALARMS, resume as resume_config,
    exiger_secrets,
)
from db import insert_measurement, insert_alarm, close_connection
from detecteur import DetecteurPV



# Même seuil que le notebook : seules les mesures diurnes (GTI > 50 W/m²) sont scorées.
# Sans ce filtre, un flux continu enverrait des données nocturnes aux modèles.
FILTRE_GTI_MIN = 50





class TamponGlissant:
    """Accumule les mesures récentes pour reconstruire une séquence LSTM.

    Le tampon est vidé dès qu'un écart dépasse gap_minutes : sinon on fabriquerait
    des transitions soir -> matin qui n'existent pas, la nuit n'étant pas mesurée.
    Chaque point garde son offset Kafka, pour qu'un redémarrage reconstruise la fenêtre.
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

    def offset_a_commiter(self, offset_courant: int) -> int:
        """Prochain offset à relire après un redémarrage (convention Kafka).

        On ne garde que les seq_len - 1 derniers points : c'est exactement ce dont
        la PROCHAINE fenêtre a besoin. Le point courant, déjà scoré, n'est donc pas
        rescoré au redémarrage. Tampon vide : offset_courant + 1.
        """
        a_garder = list(self.points)[-(self.seq_len - 1):] if self.seq_len > 1 else []
        return a_garder[0][0] if a_garder else offset_courant + 1


def scorer(detecteur, fenetre):
    """Retourne (score, anomalie, raison_non_score). raison_non_score est None si scoré."""
    if fenetre is None:
        return None, None, "fenêtre incomplète"
    try:
        resultat = detecteur.predire_lot(fenetre)
    except ValueError as e:
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
    try:
        horodatage = pd.Timestamp(mesure["time"])
        if pd.isna(horodatage):
            raise ValueError("horodatage vide")
    except (KeyError, TypeError, ValueError) as e:
        # Sans horodatage, la mesure ne peut être ni rangée ni scorée : on la signale
        # sans faire planter le consumer.
        return dict(horodatage=None, score=None, anomalie=None,
                    raison=f"horodatage invalide ({e})",
                    offset_commit=tampon.offset_a_commiter(offset))

    gti = mesure.get("GTI")
    if gti is None or pd.isna(gti) or not gti > FILTRE_GTI_MIN:
        return dict(horodatage=horodatage, score=None, anomalie=None,
                    raison="nuit ou irradiance absente",
                    offset_commit=tampon.offset_a_commiter(offset))

    fenetre = tampon.ajouter(offset, horodatage, {c: mesure.get(c) for c in detecteur.features})
    score, est_anomalie, raison = scorer(detecteur, fenetre)
    return dict(horodatage=horodatage, score=score, anomalie=est_anomalie,
                raison=raison, offset_commit=tampon.offset_a_commiter(offset))


def sans_nan(mesure: dict) -> dict:
    """Remplace NaN et infinis par None : JSON standard, accepté par JSONB."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in mesure.items()}


def publier_alarme(producer, mesure, score):
    """Publie l'alarme et ATTEND l'accusé de réception du broker.

    Lève une exception si la livraison échoue ou dépasse le délai : l'appelant
    ne doit alors pas committer l'offset.
    """
    future = producer.send(
        TOPIC_ALARMS,
        key=str(mesure["time"]).encode("utf-8"),
        value={**sans_nan(mesure), "score_fusion": score},
    )
    return future.get(timeout=DELAI_LIVRAISON_MS / 1000 + 5)


def traiter_et_persister(message, tampon, detecteur, consumer, producer, stats):
    """Un message de bout en bout. Ordre : base, puis Kafka (avec accusé), puis commit.

    Toute exception remonte AVANT le commit : le message sera relu au redémarrage
    (au moins une fois). La base est idempotente (ON CONFLICT) ; le topic alarms
    peut recevoir un doublon, identifiable par sa clé (l'horodatage).
    """
    mesure = message.value
    r = traiter_message(mesure, message.offset, tampon, detecteur)
    stats["lues"] += 1
    if r["score"] is not None:
        stats["scorees"] += 1
    if r["raison"] is not None:
        stats["raisons"][r["raison"]] = stats["raisons"].get(r["raison"], 0) + 1

    if r["horodatage"] is not None:
        insert_measurement({k.lower(): v for k, v in mesure.items()},
                           r["score"], r["anomalie"])

    if r["anomalie"]:
        insert_alarm(mesure, r["score"])
        publier_alarme(producer, mesure, r["score"])
        stats["alarmes"] += 1
        print(f"ALARME {r['horodatage']} | score {r['score']:.4f} "
              f"(seuil {detecteur.seuil:.4f})")

    consumer.commit({
        TopicPartition(message.topic, message.partition):
            OffsetAndMetadata(r["offset_commit"], ""),
    })


def run_consumer():
    exiger_secrets() 
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
        value_serializer=lambda v: json.dumps(v, allow_nan=False).encode("utf-8"),
        acks="all",
        enable_idempotence=True,
        delivery_timeout_ms=DELAI_LIVRAISON_MS,
        request_timeout_ms=REQUEST_TIMEOUT_MS,
    )

    arret = {"demande": False}

    def demander_arret(*_):
        print("\nArrêt demandé : fin du message en cours, puis fermeture.")
        arret["demande"] = True

    signal.signal(signal.SIGINT, demander_arret)
    signal.signal(signal.SIGTERM, demander_arret)

    print(f"Consumer en écoute sur '{TOPIC_IN}' (groupe '{GROUP_ID}')...")
    print(resume_config())
    stats = {"lues": 0, "scorees": 0, "alarmes": 0, "raisons": {}}
    try:
        while not arret["demande"]:
            for _, messages in consumer.poll(timeout_ms=1000).items():
                for message in messages:
                    traiter_et_persister(message, tampon, detecteur, consumer, producer, stats)
                    if stats["lues"] % 100 == 0:
                        print(f"{stats['lues']} mesures lues | {stats['scorees']} scorées "
                              f"| {stats['alarmes']} alarmes | {stats['raisons']}")
                    if arret["demande"]:
                        break
                if arret["demande"]:
                    break
    finally:
        # Ne jamais masquer l'exception d'origine par une erreur de fermeture.
        for fermer in (lambda: producer.close(timeout=10), consumer.close, close_connection):
            try:
                fermer()
            except Exception as e:
                print(f"Erreur à la fermeture (ignorée) : {e!r}")


if __name__ == "__main__":
    run_consumer()
