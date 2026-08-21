-- Migration 01 — ajout des colonnes va et ia à measurements.
--
-- init.sql n'est exécuté qu'à la CRÉATION du volume PostgreSQL : sur une base
-- déjà initialisée, il faut appliquer ce script à la main. Les six features du
-- modèle (GTI, Pg, Va, Vg, Ia, TPV) doivent toutes exister pour que la table
-- puisse alimenter DetecteurPV.
--
--   docker compose exec -T timescaledb \
--     psql -U pfa -d photovoltaique < docker/Migration1.sql
--
-- Idempotent : relançable sans risque.

ALTER TABLE measurements ADD COLUMN IF NOT EXISTS ia DOUBLE PRECISION;
ALTER TABLE measurements ADD COLUMN IF NOT EXISTS va DOUBLE PRECISION;

COMMENT ON COLUMN measurements.ia IS 'Courant onduleur (A) — feature du modèle';
COMMENT ON COLUMN measurements.va IS 'Tension onduleur (V) — feature du modèle';
COMMENT ON COLUMN measurements.anomaly_score IS
    'Score de fusion AE+LSTM dans [0,1] ; anomalie si >= config.json/seuil_fusion';
