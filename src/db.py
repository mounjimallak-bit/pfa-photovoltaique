import json
import math
import psycopg2
from psycopg2.extras import Json
from config import (PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD,
                    PG_CONNECT_TIMEOUT)

DB_CONFIG = {
       "host": PG_HOST,
    "port": PG_PORT,
    "dbname": PG_DATABASE,
    "user": PG_USER,
    "password": PG_PASSWORD,
    # Sans délai, une base injoignable bloque le consumer indéfiniment ; Kafka
    # l'exclut alors du groupe (max_poll_interval_ms) et le commit échoue.
    "connect_timeout": PG_CONNECT_TIMEOUT,
    "options": "-c statement_timeout=10000",
   
}
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


def _sans_nan(mesure: dict) -> dict:
    """JSONB refuse NaN et Infinity : on les remplace par null."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in mesure.items()}


def insert_measurement(mesure: dict, anomaly_score=None, is_anomaly=None):
    params = {c: mesure.get(c) for c in COLONNES_CAPTEUR}
    params["time"] = mesure.get("time")
    params["score"] = anomaly_score
    # bool(None) vaut False : sans ce garde-fou, le « je ne sais pas » serait
    # reconverti en « pas une anomalie » avant même d'atteindre la base.
    params["anomaly"] = None if is_anomaly is None else bool(is_anomaly)

    with get_connection().cursor() as cur:
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
                anomaly_score = COALESCE(EXCLUDED.anomaly_score,
                                         measurements.anomaly_score),
                is_anomaly    = COALESCE(EXCLUDED.is_anomaly,
                                         measurements.is_anomaly)
            """,
            params,
        )


def insert_alarm(mesure: dict, score: float):
    """Insère une alarme dans alarms (idempotente, comme les mesures)."""
    details = Json(_sans_nan(mesure), dumps=lambda o: json.dumps(o, allow_nan=False))
    with get_connection().cursor() as cur:
        cur.execute(
            """
            INSERT INTO alarms (time, alarm_type, score, details)
            VALUES (%(time)s, 'anomaly_detected', %(score)s, %(details)s)
            ON CONFLICT (time, alarm_type) DO UPDATE SET
                score = EXCLUDED.score, details = EXCLUDED.details
            """,
            {"time": mesure.get("time"), "score": score, "details": details},
        )
