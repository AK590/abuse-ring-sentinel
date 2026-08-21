-- schema.sql: Feature store + label-lag aware training view
-- Target: Postgres (production) / SQLite (hackathon mock)

-- Offline features written by the R-GCN batch job
CREATE TABLE IF NOT EXISTS offline_features (
    user_id      TEXT PRIMARY KEY,
    ring_risk_score  REAL NOT NULL DEFAULT 0.1,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_offline_user ON offline_features(user_id);

-- Batch job metadata for staleness monitoring
CREATE TABLE IF NOT EXISTS batch_metadata (
    id         INTEGER PRIMARY KEY DEFAULT 1,
    last_run   REAL NOT NULL,
    runtime_seconds REAL,
    users_scored    INTEGER
);

-- Training data view that enforces label-lag cutoff
-- Usage: SELECT * FROM training_data_safe;
-- The 60-day window is hardcoded here for clarity; in practice
-- parameterize via the application layer.
CREATE VIEW IF NOT EXISTS training_data_safe AS
SELECT t.*
FROM   transactions t
WHERE  t.timestamp <= (
    SELECT MAX(timestamp) FROM transactions
) - INTERVAL '60 days';
