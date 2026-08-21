
import os

import psycopg2
from psycopg2.extras import Json

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432)),
    "dbname": os.getenv("PGDATABASE", "photovoltaique"),
    "user": os.getenv("PGUSER", "pfa"),
    "password": os.getenv("PGPASSWORD", "pfa2026"),
}

# Colonnes capteur de measurements, dans l'ordre du schéma init.sql. Le flux
# Kafka transporte d'autres champs (Eg, Fault) : on ne retient que celles-ci,
# et une colonne absente du message devient NULL au lieu de faire échouer
# l'insertion sur un KeyError.
COLONNES_CAPTEUR = ("gti", "dti", "ta", "tpv", "pg", "ia", "ig", "va", "vg", "fg")

_conn = None


def get_connection():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(**DB_CONFIG)
        _conn.autocommit = True
    return _conn


def close_connection():
    global _conn
    if _conn is not None and not _conn.closed:
        _conn.close()
    _conn = None


def insert_measurement(mesure: dict, anomaly_score=None, is_anomaly: bool = False):
    params = {c: mesure.get(c) for c in COLONNES_CAPTEUR}
    params["time"] = mesure.get("time")
    params["score"] = anomaly_score
    params["anomaly"] = bool(is_anomaly)

    cur = get_connection().cursor()
    cur.execute(
        """
        INSERT INTO measurements
            (time, gti, dti, ta, tpv, pg, ia, ig, va, vg, fg,
             anomaly_score, is_anomaly)
        VALUES
            (%(time)s, %(gti)s, %(dti)s, %(ta)s, %(tpv)s, %(pg)s, %(ia)s,
             %(ig)s, %(va)s, %(vg)s, %(fg)s, %(score)s, %(anomaly)s)
        ON CONFLICT (time) DO UPDATE SET
            gti = EXCLUDED.gti, dti = EXCLUDED.dti, ta  = EXCLUDED.ta,
            tpv = EXCLUDED.tpv, pg  = EXCLUDED.pg,  ia  = EXCLUDED.ia,
            ig  = EXCLUDED.ig,  va  = EXCLUDED.va,  vg  = EXCLUDED.vg,
            fg  = EXCLUDED.fg,
            anomaly_score = EXCLUDED.anomaly_score,
            is_anomaly    = EXCLUDED.is_anomaly
        """,
        params,
    )
    cur.close()


def insert_alarm(mesure: dict, score: float):
    cur = get_connection().cursor()
    cur.execute(
        """
        INSERT INTO alarms (time, alarm_type, score, details)
        VALUES (%(time)s, 'anomaly_detected', %(score)s, %(details)s)
        ON CONFLICT (time, alarm_type) DO UPDATE SET
            score = EXCLUDED.score, details = EXCLUDED.details
        """,
        {"time": mesure.get("time"), "score": score, "details": Json(mesure)},
    )
    cur.close()
