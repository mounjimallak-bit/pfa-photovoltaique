-- Activer l'extension TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Table principale : mesures capteurs
--
-- Les 6 colonnes marquées [F] sont les features du modèle final
-- (model_final/config.json : GTI, Pg, Va, Vg, Ia, TPV). Elles doivent TOUTES
-- être présentes, sans quoi la table ne peut pas alimenter DetecteurPV.
CREATE TABLE measurements (
    time        TIMESTAMPTZ NOT NULL,
    gti         DOUBLE PRECISION,  -- [F] irradiation globale (W/m²)
    dti         DOUBLE PRECISION,  --     irradiation diffuse
    ta          DOUBLE PRECISION,  --     température ambiante (°C)
    tpv         DOUBLE PRECISION,  -- [F] température panneau (°C)
    pg          DOUBLE PRECISION,  -- [F] puissance produite (W)
    ia          DOUBLE PRECISION,  -- [F] courant onduleur (A)
    ig          DOUBLE PRECISION,  --     courant réseau (A)
    va          DOUBLE PRECISION,  -- [F] tension onduleur (V)
    vg          DOUBLE PRECISION,  -- [F] tension réseau (V)
    fg          DOUBLE PRECISION,  --     fréquence (Hz)
    -- Score de fusion AE+LSTM (moyenne des rangs contre la référence figée),
    -- dans [0,1]. Anomalie si >= config.json/seuil_fusion.
    anomaly_score DOUBLE PRECISION,
    is_anomaly  BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('measurements', 'time');

-- Unicité temporelle : le consumer insère en ON CONFLICT (time), de sorte que
-- rejouer la démo écrase les lignes au lieu de les dupliquer. TimescaleDB
-- impose que l'index unique contienne la colonne de partitionnement.
CREATE UNIQUE INDEX measurements_time_uniq ON measurements (time);

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

CREATE UNIQUE INDEX alarms_time_type_uniq ON alarms (time, alarm_type);

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
