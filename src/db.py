"""
Connexion et insertion dans TimescaleDB.
"""
import psycopg2
from psycopg2.extras import execute_values

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "photovoltaique",
    "user": "pfa",
    "password": "pfa2026",
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def insert_measurement(mesure: dict, anomaly_score: float, is_anomaly: bool):
    """Insère une mesure + score dans la table measurements."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO measurements (time, gti, dti, ta, tpv, pg, ig, vg, fg, anomaly_score, is_anomaly)
        VALUES (%(time)s, %(gti)s, %(dti)s, %(ta)s, %(tpv)s, %(pg)s, %(ig)s, %(vg)s, %(fg)s, %(score)s, %(anomaly)s)
        """,
        {**mesure, "score": anomaly_score, "anomaly": is_anomaly},
    )
    conn.commit()
    cur.close()
    conn.close()


def insert_alarm(mesure: dict, score: float):
    """Insère une alarme dans la table alarms."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO alarms (time, alarm_type, score, details)
        VALUES (%(time)s, 'anomaly_detected', %(score)s, %(details)s)
        """,
        {"time": mesure.get("time"), "score": score, "details": psycopg2.extras.Json(mesure)},
    )
    conn.commit()
    cur.close()
    conn.close()
