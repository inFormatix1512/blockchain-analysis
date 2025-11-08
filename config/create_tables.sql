-- Table for periodic mempool snapshots
CREATE TABLE IF NOT EXISTS mempool_snapshot (
  id SERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  total_tx INTEGER,
  total_size INTEGER,
  total_fee BIGINT,
  raw JSONB
);

-- Index per query temporali veloci
CREATE INDEX IF NOT EXISTS idx_mempool_snapshot_ts ON mempool_snapshot(ts);

-- Basic transaction information
CREATE TABLE IF NOT EXISTS tx_basic (
  txid TEXT PRIMARY KEY,
  block_height INTEGER,
  ts TIMESTAMPTZ,
  size INTEGER,
  vsize INTEGER,
  weight INTEGER,
  fee BIGINT,
  feerate NUMERIC,
  inputs_count INTEGER,
  outputs_count INTEGER,
  script_types JSONB,
  raw JSONB
);

-- Indici per performance
CREATE INDEX IF NOT EXISTS idx_tx_basic_block_height ON tx_basic(block_height);
CREATE INDEX IF NOT EXISTS idx_tx_basic_ts ON tx_basic(ts);
CREATE INDEX IF NOT EXISTS idx_tx_basic_feerate ON tx_basic(feerate) WHERE feerate IS NOT NULL;

-- Heuristics results per transaction
CREATE TABLE IF NOT EXISTS tx_heuristics (
  txid TEXT PRIMARY KEY REFERENCES tx_basic(txid),
  is_rbf BOOLEAN,
  coinjoin_score NUMERIC,
  equal_output BOOLEAN,
  likely_change_index INTEGER,
  notes TEXT
);

-- Indici per query analitiche
CREATE INDEX IF NOT EXISTS idx_tx_heuristics_rbf ON tx_heuristics(is_rbf);
CREATE INDEX IF NOT EXISTS idx_tx_heuristics_equal_output ON tx_heuristics(equal_output);
CREATE INDEX IF NOT EXISTS idx_tx_heuristics_coinjoin ON tx_heuristics(coinjoin_score) WHERE coinjoin_score > 0.5;

-- Aggregated monthly statistics (example table for further analysis)
CREATE TABLE IF NOT EXISTS stats_monthly (
  month DATE PRIMARY KEY,
  avg_feerate NUMERIC,
  median_confirm_blocks INTEGER,
  rbf_rate NUMERIC,
  coinjoin_pct NUMERIC
);
