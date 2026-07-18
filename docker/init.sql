-- Activer l'extension TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Table principale : mesures capteurs
CREATE TABLE measurements (
    time        TIMESTAMPTZ NOT NULL,
    gti         DOUBLE PRECISION,  -- irradiation globale (W/m²)
    dti         DOUBLE PRECISION,  -- irradiation diffuse
    ta          DOUBLE PRECISION,  -- température ambiante (°C)
    tpv         DOUBLE PRECISION,  -- température panneau (°C)
    pg          DOUBLE PRECISION,  -- puissance produite (W)
    ig          DOUBLE PRECISION,  -- courant réseau (A)
    vg          DOUBLE PRECISION,  -- tension réseau (V)
    fg          DOUBLE PRECISION,  -- fréquence (Hz)
    anomaly_score DOUBLE PRECISION, -- score du modèle ML
    is_anomaly  BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('measurements', 'time');

-- Table des alarmes
CREATE TABLE alarms (
    id          SERIAL,
    time        TIMESTAMPTZ NOT NULL,
    alarm_type  VARCHAR(100),       -- type détecté par le modèle
    severity    VARCHAR(20) DEFAULT 'warning',
    score       DOUBLE PRECISION,
    details     JSONB,
    resolved    BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('alarms', 'time');

-- Table historique maintenance
CREATE TABLE maintenance_actions (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    alarm_id    INTEGER,
    action_type VARCHAR(100),
    description TEXT,
    status      VARCHAR(20) DEFAULT 'pending'
);

-- Index utiles
CREATE INDEX idx_measurements_anomaly ON measurements (time, is_anomaly)
    WHERE is_anomaly = TRUE;
CREATE INDEX idx_alarms_unresolved ON alarms (time, resolved)
    WHERE resolved = FALSE;
