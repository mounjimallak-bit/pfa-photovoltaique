-- Migration 02 — unicité temporelle sur measurements et alarms.
--
-- Sans contrainte d'unicité, relancer la démo réinsérait les mêmes points :
-- le consumer repartait de l'offset 0 du topic Kafka et TimescaleDB acceptait
-- chaque doublon. Grafana affichait alors deux séries superposées.
--
-- Le consumer déclare désormais un group_id (il reprend où il s'était arrêté)
-- ET les insertions passent par ON CONFLICT : la ligne est mise à jour plutôt
-- que dupliquée. Ce second garde-fou exige les index ci-dessous.
--
-- TimescaleDB impose que tout index unique contienne la colonne de
-- partitionnement (time) : c'est le cas ici.
--
--   docker compose exec -T timescaledb \
--     psql -U pfa -d photovoltaique < docker/Migration2.sql
--
-- Idempotent : relançable sans risque. À appliquer APRÈS Migration1.sql.

-- Purge des doublons éventuels, en gardant la ligne la plus récemment écrite.
DELETE FROM measurements a USING measurements b
WHERE a.ctid < b.ctid AND a.time = b.time;

DELETE FROM alarms a USING alarms b
WHERE a.ctid < b.ctid AND a.time = b.time AND a.alarm_type = b.alarm_type;

CREATE UNIQUE INDEX IF NOT EXISTS measurements_time_uniq
    ON measurements (time);

CREATE UNIQUE INDEX IF NOT EXISTS alarms_time_type_uniq
    ON alarms (time, alarm_type);
