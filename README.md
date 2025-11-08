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

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 10GB disk space (for pruned node)
- Python 3.9+ (for analysis notebooks)

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

Edit `config/bitcoin.conf` to customize Bitcoin Core:

```conf
prune=10000        # Keep last 10GB of blocks
rpcuser=bitcoin
rpcpassword=bitcoin
```

Edit `.env` for database credentials and RPC settings.

## Performance

- **Query Latency**: <10ms (99th percentile)
- **Throughput**: ~50 transactions/minute (with 120s polling)
- **Resource Usage**: ~15% CPU, ~500MB RAM
- **Storage**: 10GB (pruned node) + ~1GB database (for 50k transactions)

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

- **Proof-of-concept**: Tested on genesis blocks (50 transactions)
- **Simplified heuristics**: Production systems use more sophisticated algorithms
- **No ground truth**: ML models lack labeled training data
- **Pruned node**: Cannot query arbitrary historical transactions

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
