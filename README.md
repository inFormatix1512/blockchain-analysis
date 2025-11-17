# Bitcoin Chain Analysis System

A complete system for Bitcoin blockchain analysis using chain analysis techniques, privacy heuristics, and machine learning models.

## What It Does

This project implements a **proof-of-concept** blockchain analysis system that:

- **Collects blockchain data** from a Bitcoin Core node (pruned mode) via RPC
- **Stores transactions** in a PostgreSQL database with computed heuristics
- **Applies privacy heuristics** to identify patterns:
  - Replace-By-Fee (RBF) detection
  - Equal output detection (CoinJoin indicator)
  - CoinJoin scoring algorithm
  - Change address identification
- **Implements ML models** for automated classification:
  - CoinJoin transaction detector (Random Forest)
  - Fee rate predictor (Gradient Boosting)
  - Transaction clusterer (K-Means)
  - Anomaly detector (Isolation Forest)
- **Monitors mempool** for real-time network analysis
- **Provides forensic analysis** capabilities (address clustering, taint analysis)

## Architecture

```
Bitcoin Core (pruned) → Ingest Service (Python) → PostgreSQL → Analysis/ML
```

**Tech Stack**: Docker, Bitcoin Core, PostgreSQL, Python, Pandas, Scikit-learn, Jupyter

### Pruned Node Configuration

The system uses **Bitcoin Core in pruned mode** to minimize disk usage while maintaining full blockchain validation:

- **Prune size**: 140GB (configurable via `prune=140000` in `bitcoin.conf`)
- **Mode**: Full node validation with automatic old block pruning
- **Retention**: Keeps last ~140,000 MB of blocks (~2.5 years of blockchain data)
- **Benefits**: 
  - Full security (validates entire blockchain)
  - Balanced storage (140GB vs ~600GB for full archival node)
  - Sufficient for multi-year historical analysis
- **Trade-off**: Cannot query arbitrary historical blocks older than retention window

### Auto-Stop Ingest Mechanism

The ingest service features **automatic completion** when reaching data collection targets:

- **Target height**: Configurable via `INGEST_END_HEIGHT` environment variable (default: 600,188 blocks)
- **Auto-stop**: Service monitors database progress and gracefully exits when target is reached
- **Progress tracking**: Real-time logging with percentage completion
- **Example log output**:
  ```
  📊 Progresso: 450000/600188 (74.98%)
  🎯 OBIETTIVO RAGGIUNTO! Blocco 600188/600188
  ✅ Ingestione completata. Container si arresta.
  ```
- **Use case**: Collect specific historical datasets (e.g., May 2017 - October 2019) without manual intervention

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 160GB+ disk space (140GB pruned blockchain + 20GB database)
- Python 3.9+ (for analysis notebooks)
- 8GB+ RAM recommended
- Stable internet connection for blockchain synchronization

### 1. Start the System

```bash
docker-compose up -d
```

This will start:
- **bitcoind**: Bitcoin Core node (syncs blockchain in pruned mode)
- **postgres**: PostgreSQL database
- **ingest**: Python service that processes blocks and computes heuristics

### 2. Wait for Initial Sync

Bitcoin Core needs to sync with the network (can take hours/days depending on your connection). Monitor progress:

```bash
docker-compose logs -f bitcoind
```

### 3. Access the Database

Connect to PostgreSQL to query collected data:

```bash
docker exec -it postgres psql -U postgres -d blockchain
```

Example queries:

```sql
-- Count transactions
SELECT COUNT(*) FROM tx_basic;

-- Check RBF transactions
SELECT COUNT(*) FROM tx_heuristics WHERE is_rbf = true;

-- CoinJoin candidates
SELECT * FROM tx_heuristics WHERE coinjoin_score > 0.7;
```

### 4. Run Analysis

Activate Python environment and run Jupyter notebooks:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt

jupyter notebook analysis/analysis.ipynb
```

## Database Schema

### Tables

**`tx_basic`**: Core transaction data (txid, block height, size, fee, I/O counts)

**`tx_heuristics`**: Privacy heuristics (is_rbf, equal_output, coinjoin_score, change_address_idx)

**`mempool_snapshot`**: Periodic mempool state (tx count, total size, total fees)

## Machine Learning Models

Train models on collected data:

```bash
python ml/train_models.py
```

This trains 4 models:
1. **CoinJoin Detector**: Classifies mixing transactions
2. **Fee Rate Predictor**: Estimates optimal fees for fast confirmation
3. **Transaction Clusterer**: Groups transactions by behavioral patterns
4. **Anomaly Detector**: Flags suspicious transactions

Use trained models for predictions:

```bash
python ml/predict.py
```

## Testing

Run system tests:

```bash
python scripts/tests/test_system.py
python scripts/tests/test_ml.py
```

## Configuration

### Bitcoin Core Settings

Edit `config/bitcoin.conf` to customize Bitcoin Core:

```conf
prune=140000       # Keep last 140GB of blocks (~2.5 years of data)
rpcuser=bitcoin
rpcpassword=bitcoin123
server=1           # Enable RPC server
txindex=0          # Disabled (not needed in pruned mode)
rpcbind=0.0.0.0    # Bind to all interfaces (Docker networking)
rpcallowip=172.18.0.0/16  # Allow Docker subnet
```

### Ingest Configuration

Edit `.env` file to configure data collection parameters:

```env
# Bitcoin RPC
RPC_USER=bitcoin
RPC_PASSWORD=bitcoin123
RPC_HOST=bitcoind
RPC_PORT=8332

# Database
PGHOST=postgres
PGUSER=postgres
PGPASSWORD=postgres
PGDATABASE=blockchain

# Ingest settings
INGEST_END_HEIGHT=600188          # Auto-stop at this block height (Oct 2019)
MAX_BLOCKS_PER_RUN=1000           # Blocks processed per cycle (high throughput)
BLOCK_INGEST_INTERVAL=30          # Seconds between ingest cycles
MEMPOOL_SNAPSHOT_INTERVAL=60      # Seconds between mempool snapshots
```

### Database Configuration

Default credentials (change for production):
- **User**: `postgres`
- **Password**: `postgres`
- **Database**: `blockchain`

**Security Note**: These are development defaults. For production deployments, use strong passwords and restrict network access (bind PostgreSQL to 127.0.0.1 only).

## Performance

- **Query Latency**: <10ms (99th percentile)
- **Throughput**: ~1,400 blocks/hour during ingest, processing ~2,500 transactions/minute
- **Resource Usage**: 
  - CPU: ~150% bitcoind (verification), ~35% python ingest, ~40-80% postgres (under load)
  - RAM: ~1.6GB (bitcoind), ~150MB (postgres), ~80MB (ingest)
- **Storage**: 
  - Bitcoin data: ~140GB (pruned mode with 140GB retention)
  - Database: ~800MB per 2M transactions (without raw JSONB)
  - Total recommended: 160GB disk minimum
- **Sync Time**: 
  - Full sync to block 657k: ~24 hours (depends on network speed)
  - Ingest rate: ~1,400 blocks/hour (1000 blocks per 30s cycle)
  - Complete ingest (552k-600k): ~33 hours for 47k blocks

## Use Cases

### Forensic Analysis
- Track funds through the blockchain
- Identify wallet clustering patterns
- Detect mixing/tumbling attempts

### Privacy Research
- Evaluate effectiveness of privacy tools (CoinJoin, RBF)
- Analyze transaction patterns
- Study deanonymization techniques

### Network Monitoring
- Real-time mempool analysis
- Fee rate estimation
- Anomaly detection

## Limitations

- **Pruned node constraints**: Cannot query blocks older than retention window (~140GB / ~47k blocks)
- **Historical data collection**: Requires full sync beyond target height before ingest can process blocks
- **Simplified heuristics**: Production systems use more sophisticated algorithms
- **ML training data**: Models lack comprehensive labeled datasets for validation
- **Resource requirements**: Requires stable infrastructure for multi-day synchronization
- **Network dependency**: Initial blockchain download requires reliable high-speed internet

### Important Note on Pruned Mode

When using pruned mode with `prune=140000`:
- Bitcoin Core downloads and validates **entire blockchain** sequentially
- Only the **last 140GB** of blocks are retained on disk
- Older blocks are automatically deleted after validation
- **Ingest service** can only process blocks currently available in retention window
- Current dataset: blocks 552,905-600,188 (May 2018 - Oct 2019, ~47k blocks)
- To collect blocks up to height N: Bitcoin must sync significantly past N (so target range falls within retention window)

**Strategy**: Configure `INGEST_END_HEIGHT` based on your pruned retention size and target dataset.

**Disk Space Optimization**: If running low on disk space, you can disable the `raw` JSONB column in `block_ingest.py` (comment out the raw column insertion). This reduces database size by ~70% but disables fee calculation for new blocks. Fee data will still be available for blocks ingested before the optimization.

## Extending the System

### Add New Heuristics

Edit `ingest/block_ingest.py` and implement in `calculate_heuristics()`:

```python
def calculate_heuristics(tx):
    return {
        'is_rbf': check_rbf(tx),
        'your_new_heuristic': your_function(tx),
        # ...
    }
```

### Add ML Models

Create new model class in `ml/train_models.py`:

```python
class YourModel:
    def __init__(self):
        self.model = YourSklearnModel()
    
    def train(self, df):
        # Training logic
        pass
```

### Multi-Chain Support

Adapt RPC calls and transaction parsing for other UTXO-based chains (Litecoin, Dogecoin, etc.).

## License

This project is for educational and research purposes. Use responsibly and ethically.

## Citation

If you use this work, please cite:

```
Impellizzeri, L. (2025). Bitcoin Chain Analysis System: 
Data Mining Techniques and Forensic Applications. 
University of Catania.
```

## Disclaimer

This tool is intended for legitimate research and educational purposes only. Chain analysis can compromise user privacy - use responsibly and in compliance with applicable laws.
