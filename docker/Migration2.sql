
DELETE FROM measurements a USING measurements b
WHERE a.ctid < b.ctid AND a.time = b.time;

DELETE FROM alarms a USING alarms b
WHERE a.ctid < b.ctid AND a.time = b.time AND a.alarm_type = b.alarm_type;

CREATE UNIQUE INDEX IF NOT EXISTS measurements_time_uniq
    ON measurements (time);

CREATE UNIQUE INDEX IF NOT EXISTS alarms_time_type_uniq
    ON alarms (time, alarm_type);
