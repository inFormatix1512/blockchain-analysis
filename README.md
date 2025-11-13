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

- **Prune size**: 50GB (configurable via `prune=50000` in `bitcoin.conf`)
- **Mode**: Full node validation with automatic old block pruning
- **Retention**: Keeps last ~50,000 blocks available for queries
- **Benefits**: 
  - Full security (validates entire blockchain)
  - Minimal storage (~50GB vs ~600GB for full archival node)
  - Suitable for resource-constrained environments
- **Trade-off**: Cannot query arbitrary historical blocks older than retention window

### Auto-Stop Ingest Mechanism

The ingest service features **automatic completion** when reaching data collection targets:

- **Target height**: Configurable via `INGEST_END_HEIGHT` environment variable (default: 150,000 blocks)
- **Auto-stop**: Service monitors database progress and gracefully exits when target is reached
- **Progress tracking**: Real-time logging with percentage completion
- **Example log output**:
  ```
  📊 Progresso: 75000/150000 (50.00%)
  🎯 OBIETTIVO RAGGIUNTO! Blocco 150000/150000
  ✅ Ingestione completata. Container si arresta.
  ```
- **Use case**: Collect specific historical datasets (e.g., early Bitcoin era 2009-2011) without manual intervention

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 50GB+ disk space (for pruned node with 50GB prune size)
- Python 3.9+ (for analysis notebooks)
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

## Project Structure

```
├── docker-compose.yml          # Container orchestration
├── config/
│   ├── bitcoin.conf            # Bitcoin Core configuration
│   └── create_tables.sql       # Database schema
├── ingest/
│   ├── block_ingest.py         # Block processing logic
│   ├── mempool_snapshot.py     # Mempool monitoring
│   └── run_ingest.py           # Main ingestion loop
├── ml/
│   ├── train_models.py         # ML model training
│   └── predict.py              # Inference scripts
├── analysis/
│   ├── analysis.ipynb          # Data exploration notebook
│   └── ml_analysis.ipynb       # ML analysis notebook
└── scripts/
    └── tests/                  # System tests
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
prune=50000        # Keep last 50GB of blocks (maintains ~50,000 blocks)
rpcuser=bitcoinrpc
rpcpassword=secure_rpc_password_2024
server=1           # Enable RPC server
txindex=0          # Disabled (not needed in pruned mode)
```

### Ingest Configuration

Edit `.env` file to configure data collection parameters:

```env
# Bitcoin RPC
BITCOIN_RPC_USER=bitcoinrpc
BITCOIN_RPC_PASSWORD=secure_rpc_password_2024
BITCOIN_RPC_HOST=bitcoind
BITCOIN_RPC_PORT=8332

# Database
POSTGRES_USER=blockchain_user
POSTGRES_PASSWORD=blockchain_secure_password_2024
POSTGRES_DB=blockchain_analysis

# Ingest settings
INGEST_END_HEIGHT=150000          # Auto-stop at this block height
MAX_BLOCKS_PER_RUN=500            # Blocks processed per cycle
BLOCK_INGEST_INTERVAL=60          # Seconds between ingest cycles
MEMPOOL_SNAPSHOT_INTERVAL=300     # Seconds between mempool snapshots
```

### Database Configuration

Default credentials (change for production):
- **User**: `blockchain_user`
- **Password**: `blockchain_secure_password_2024`
- **Database**: `blockchain_analysis`

## Performance

- **Query Latency**: <10ms (99th percentile)
- **Throughput**: ~500 blocks/hour during sync, ~50 transactions/minute sustained
- **Resource Usage**: 
  - CPU: ~15-30% (during sync), ~5% (idle)
  - RAM: ~1.5GB (bitcoind), ~500MB (postgres), ~256MB (ingest)
- **Storage**: 
  - Bitcoin data: ~50GB (pruned mode)
  - Database: ~1GB per 100k transactions
  - Total recommended: 80GB disk minimum
- **Sync Time**: 
  - First 150k blocks: 48-72 hours (depends on network speed)
  - Ingest rate: ~2,000-3,000 blocks/hour during peak

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

- **Pruned node constraints**: Cannot query blocks older than retention window (~50,000 blocks)
- **Historical data collection**: Requires full sync to target height before ingest can process blocks
- **Simplified heuristics**: Production systems use more sophisticated algorithms
- **ML training data**: Models lack comprehensive labeled datasets for validation
- **Resource requirements**: Requires stable infrastructure for multi-day synchronization
- **Network dependency**: Initial blockchain download requires reliable high-speed internet

### Important Note on Pruned Mode

When using pruned mode with `prune=50000`:
- Bitcoin Core downloads and validates **entire blockchain** sequentially
- Only the **last 50GB** (~50,000 most recent blocks) are retained on disk
- Older blocks are automatically deleted after validation
- **Ingest service** can only process blocks currently available in retention window
- To collect blocks 0-150,000: Bitcoin must sync past block ~200,000 (so blocks 150k-200k are in the 50k window)

**Strategy**: Configure `INGEST_END_HEIGHT` based on your pruned retention size and target dataset.

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
