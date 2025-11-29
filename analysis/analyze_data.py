#!/usr/bin/env python3
"""
Exploratory data analysis for blockchain data.

Provides quick analysis of collected transaction and mempool data
as an alternative to Jupyter notebooks.
"""

import sys
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from common import get_db_connection

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
except ImportError:
    HAS_VISUALIZATION = False


class BlockchainAnalyzer:
    """
    Analyzer for blockchain transaction data.
    
    Provides methods for exploratory data analysis including
    transaction statistics, block analysis, and heuristics summary.
    """
    
    SECTION_WIDTH = 70
    
    def __init__(self):
        """Initialize analyzer."""
        # Fix encoding for Windows console
        if sys.platform == 'win32':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except AttributeError:
                pass
    
    def _print_header(self, title: str) -> None:
        """Print formatted section header."""
        print("\n" + "=" * self.SECTION_WIDTH)
        print(f"  {title}")
        print("=" * self.SECTION_WIDTH + "\n")
    
    def _print_subheader(self, title: str) -> None:
        """Print formatted subsection header."""
        print(f"\n{title}")
        print("-" * self.SECTION_WIDTH)
    
    def load_transactions(self) -> pd.DataFrame:
        """
        Load transaction data from database.
        
        Returns:
            DataFrame with transaction and heuristics data.
        """
        query = """
            SELECT 
                t.txid,
                t.block_height,
                t.ts,
                t.size,
                t.vsize,
                t.weight,
                t.fee,
                t.feerate,
                t.inputs_count,
                t.outputs_count,
                h.is_rbf,
                h.coinjoin_score,
                h.equal_output
            FROM tx_basic t
            LEFT JOIN tx_heuristics h ON t.txid = h.txid
            ORDER BY t.block_height
        """
        
        with get_db_connection() as conn:
            return pd.read_sql(query, conn)
    
    def load_mempool_snapshots(self, limit: int = 20) -> pd.DataFrame:
        """
        Load recent mempool snapshots.
        
        Args:
            limit: Maximum number of snapshots to load.
            
        Returns:
            DataFrame with mempool snapshot data.
        """
        query = f"""
            SELECT 
                id,
                ts,
                total_tx,
                total_size,
                total_fee
            FROM mempool_snapshot
            ORDER BY ts DESC
            LIMIT {limit}
        """
        
        with get_db_connection() as conn:
            return pd.read_sql(query, conn)
    
    def analyze_transactions(self) -> Optional[pd.DataFrame]:
        """
        Perform comprehensive transaction analysis.
        
        Returns:
            DataFrame with transaction data, or None on failure.
        """
        self._print_header("TRANSACTION ANALYSIS")
        
        df = self.load_transactions()
        
        print(f"[INFO] Dataset: {len(df):,} transactions loaded\n")
        
        if df.empty:
            print("[WARN] No transactions found in database")
            return None
        
        # Descriptive statistics
        self._print_subheader("DESCRIPTIVE STATISTICS")
        stats_cols = ['block_height', 'size', 'inputs_count', 'outputs_count', 'fee']
        print(df[stats_cols].describe().round(2))
        
        # Block analysis
        self._print_subheader("BLOCK ANALYSIS")
        blocks = df.groupby('block_height').agg({
            'txid': 'count',
            'size': 'mean',
            'fee': 'sum',
            'inputs_count': 'mean',
            'outputs_count': 'mean'
        }).round(2)
        blocks.columns = ['TX Count', 'Avg Size', 'Total Fees', 'Avg Inputs', 'Avg Outputs']
        print(blocks.tail(10))
        
        # Heuristics analysis
        self._print_subheader("HEURISTICS ANALYSIS")
        rbf_count = df['is_rbf'].sum()
        equal_output_count = df['equal_output'].sum()
        total = len(df)
        
        print(f"Transactions with RBF: {rbf_count:,} ({rbf_count/total*100:.1f}%)")
        print(f"Transactions with equal outputs: {equal_output_count:,} ({equal_output_count/total*100:.1f}%)")
        print(f"Average CoinJoin score: {df['coinjoin_score'].mean():.3f}")
        print(f"Maximum CoinJoin score: {df['coinjoin_score'].max():.3f}")
        
        # Input/Output distribution
        self._print_subheader("INPUT/OUTPUT DISTRIBUTION (Top 10)")
        io_dist = df.groupby(['inputs_count', 'outputs_count']).size().reset_index(name='count')
        io_dist = io_dist.sort_values('count', ascending=False).head(10)
        print(io_dist.to_string(index=False))
        
        return df
    
    def analyze_mempool(self) -> Optional[pd.DataFrame]:
        """
        Analyze mempool snapshots.
        
        Returns:
            DataFrame with mempool data, or None on failure.
        """
        self._print_header("MEMPOOL SNAPSHOTS ANALYSIS")
        
        df = self.load_mempool_snapshots()
        
        print(f"[INFO] {len(df)} snapshots loaded\n")
        
        if df.empty:
            print("[WARN] No mempool snapshots available")
            return None
        
        # Statistics
        self._print_subheader("MEMPOOL STATISTICS")
        print(df[['total_tx', 'total_size', 'total_fee']].describe().round(2))
        
        # Recent snapshots
        self._print_subheader("LAST 10 SNAPSHOTS")
        print(df[['ts', 'total_tx', 'total_size', 'total_fee']].head(10).to_string(index=False))
        
        return df
    
    def generate_summary(self) -> dict:
        """
        Generate comprehensive database summary.
        
        Returns:
            Dictionary with summary statistics.
        """
        self._print_header("GENERAL SUMMARY")
        
        summary = {}
        
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            # Count records
            cur.execute('SELECT COUNT(*) FROM tx_basic')
            summary['tx_count'] = cur.fetchone()[0]
            
            cur.execute('SELECT COUNT(*) FROM mempool_snapshot')
            summary['snapshot_count'] = cur.fetchone()[0]
            
            cur.execute('SELECT MIN(block_height), MAX(block_height) FROM tx_basic')
            summary['min_block'], summary['max_block'] = cur.fetchone()
            
            cur.execute('SELECT MIN(ts), MAX(ts) FROM tx_basic')
            summary['first_ts'], summary['last_ts'] = cur.fetchone()
            
            # Database size
            cur.execute("""
                SELECT 
                    pg_size_pretty(pg_total_relation_size('tx_basic')) as tx_size,
                    pg_size_pretty(pg_total_relation_size('tx_heuristics')) as heur_size,
                    pg_size_pretty(pg_total_relation_size('mempool_snapshot')) as snap_size
            """)
            sizes = cur.fetchone()
            summary['tx_basic_size'] = sizes[0]
            summary['tx_heuristics_size'] = sizes[1]
            summary['mempool_snapshot_size'] = sizes[2]
        
        # Print summary
        print(f"Total transactions: {summary['tx_count']:,}")
        print(f"Mempool snapshots: {summary['snapshot_count']:,}")
        print(f"Block range: {summary['min_block']} -> {summary['max_block']}")
        print(f"Time period: {summary['first_ts']} -> {summary['last_ts']}")
        
        self._print_subheader("DATABASE SIZE")
        print(f"Table tx_basic: {summary['tx_basic_size']}")
        print(f"Table tx_heuristics: {summary['tx_heuristics_size']}")
        print(f"Table mempool_snapshot: {summary['mempool_snapshot_size']}")
        
        return summary
    
    def run_full_analysis(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], dict]:
        """
        Run complete exploratory analysis.
        
        Returns:
            Tuple of (transaction_df, mempool_df, summary_dict).
        """
        print("\n" + "=" * self.SECTION_WIDTH)
        print(" " * 15 + "BLOCKCHAIN EXPLORATORY DATA ANALYSIS")
        print("=" * self.SECTION_WIDTH)
        print(f"Executed: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        
        try:
            df_tx = self.analyze_transactions()
            df_mempool = self.analyze_mempool()
            summary = self.generate_summary()
            
            self._print_header("ANALYSIS COMPLETED SUCCESSFULLY")
            
            print("NEXT STEPS:")
            print("   - Use the DataFrames for further analysis")
            print("   - Create visualizations with matplotlib/seaborn")
            print("   - Apply ML models from collected data")
            print("   - Export data: df.to_csv('export.csv')")
            print("")
            
            return df_tx, df_mempool, summary
            
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            return None, None, {}


def main() -> bool:
    """
    Execute full analysis.
    
    Returns:
        True on success, False on failure.
    """
    analyzer = BlockchainAnalyzer()
    df_tx, df_mempool, summary = analyzer.run_full_analysis()
    return df_tx is not None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
