#!/usr/bin/env python3
"""
ML prediction interface for blockchain transaction analysis.

Provides a command-line interface for running predictions
using trained ML models.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

# Add parent directory and current directory to path for imports
_parent_dir = str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0]
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _parent_dir)
sys.path.insert(0, _current_dir)

from common import get_db_connection
from train_models import (
    AnomalyDetector,
    CoinJoinDetector,
    FeeratePredictor,
    TransactionClusterer,
    engineer_features,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class TransactionAnalyzer:
    """
    Analyzer for running ML predictions on blockchain transactions.
    
    Orchestrates multiple ML models to provide comprehensive
    analysis of transaction patterns and anomalies.
    """
    
    def __init__(self):
        """Initialize analyzer with lazy model loading."""
        self._coinjoin_detector: Optional[CoinJoinDetector] = None
        self._feerate_predictor: Optional[FeeratePredictor] = None
        self._clusterer: Optional[TransactionClusterer] = None
        self._anomaly_detector: Optional[AnomalyDetector] = None
    
    @property
    def coinjoin_detector(self) -> CoinJoinDetector:
        """Lazy load CoinJoin detector."""
        if self._coinjoin_detector is None:
            self._coinjoin_detector = CoinJoinDetector.load()
        return self._coinjoin_detector
    
    @property
    def feerate_predictor(self) -> FeeratePredictor:
        """Lazy load feerate predictor."""
        if self._feerate_predictor is None:
            self._feerate_predictor = FeeratePredictor.load()
        return self._feerate_predictor
    
    @property
    def clusterer(self) -> TransactionClusterer:
        """Lazy load transaction clusterer."""
        if self._clusterer is None:
            self._clusterer = TransactionClusterer.load()
        return self._clusterer
    
    @property
    def anomaly_detector(self) -> AnomalyDetector:
        """Lazy load anomaly detector."""
        if self._anomaly_detector is None:
            self._anomaly_detector = AnomalyDetector.load()
        return self._anomaly_detector
    
    def load_recent_transactions(self, limit: int = 1000) -> pd.DataFrame:
        """
        Load recent transactions for analysis.
        
        Args:
            limit: Maximum number of transactions to load.
            
        Returns:
            DataFrame with transaction data.
        """
        logger.info("Loading last %d transactions...", limit)
        
        query = f"""
        SELECT 
            b.txid,
            b.block_height,
            EXTRACT(EPOCH FROM b.ts) as timestamp,
            b.size,
            b.vsize,
            b.weight,
            b.fee,
            b.feerate,
            b.inputs_count,
            b.outputs_count,
            h.is_rbf,
            h.coinjoin_score,
            h.equal_output,
            h.likely_change_index
        FROM tx_basic b
        JOIN tx_heuristics h ON b.txid = h.txid
        WHERE b.fee IS NOT NULL 
            AND b.feerate IS NOT NULL
            AND b.feerate > 0
        ORDER BY b.block_height DESC
        LIMIT {limit}
        """
        
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info("Loaded %d transactions", len(df))
        return df
    
    def analyze(self, limit: int = 1000) -> pd.DataFrame:
        """
        Run comprehensive ML analysis on transactions.
        
        Args:
            limit: Number of transactions to analyze.
            
        Returns:
            DataFrame with analysis results.
        """
        logger.info("=" * 50)
        logger.info("Starting ML Analysis")
        logger.info("=" * 50)
        
        # Load data
        df = self.load_recent_transactions(limit)
        
        if df.empty:
            logger.error("No transactions found")
            return pd.DataFrame()
        
        # Feature engineering
        df = engineer_features(df)
        
        # 1. CoinJoin Detection
        logger.info("\n1. CoinJoin Analysis...")
        try:
            df['coinjoin_probability'] = self.coinjoin_detector.predict(df)
            high_coinjoin = df[df['coinjoin_probability'] > 0.7]
            logger.info("   High CoinJoin probability transactions: %d", len(high_coinjoin))
            
            if not high_coinjoin.empty:
                logger.info("   Top 5 CoinJoin candidates:")
                for _, row in high_coinjoin.nlargest(5, 'coinjoin_probability').iterrows():
                    logger.info(
                        "     - %s... (prob: %.2f%%)",
                        row['txid'][:16], row['coinjoin_probability'] * 100
                    )
        except FileNotFoundError:
            logger.warning("   CoinJoin detector model not found")
            df['coinjoin_probability'] = None
        
        # 2. Feerate Prediction
        logger.info("\n2. Feerate Prediction...")
        try:
            df['predicted_feerate'] = self.feerate_predictor.predict(df)
            df['feerate_diff'] = df['feerate'] - df['predicted_feerate']
            
            logger.info("   Current avg feerate: %.2f sat/vB", df['feerate'].mean())
            logger.info("   Predicted avg feerate: %.2f sat/vB", df['predicted_feerate'].mean())
            
            std = df['feerate'].std()
            overpaying = df[df['feerate_diff'] > std]
            underpaying = df[df['feerate_diff'] < -std]
            
            logger.info("   Overpaying transactions: %d", len(overpaying))
            logger.info("   Underpaying transactions: %d", len(underpaying))
        except FileNotFoundError:
            logger.warning("   Feerate predictor model not found")
            df['predicted_feerate'] = None
            df['feerate_diff'] = None
        
        # 3. Clustering
        logger.info("\n3. Transaction Clustering...")
        try:
            X = self.clusterer.prepare_features(df)
            X_scaled = self.clusterer.scaler.transform(X)
            df['cluster'] = self.clusterer.model.predict(X_scaled)
            
            logger.info("   Cluster distribution:")
            for cluster_id in sorted(df['cluster'].unique()):
                count = (df['cluster'] == cluster_id).sum()
                logger.info(
                    "     Cluster %d: %d transactions (%.1f%%)",
                    cluster_id, count, count / len(df) * 100
                )
        except FileNotFoundError:
            logger.warning("   Clusterer model not found")
            df['cluster'] = None
        
        # 4. Anomaly Detection
        logger.info("\n4. Anomaly Detection...")
        try:
            predictions = self.anomaly_detector.predict(df)
            df['is_anomaly'] = predictions == -1
            
            n_anomalies = df['is_anomaly'].sum()
            logger.info(
                "   Anomalies detected: %d (%.1f%%)",
                n_anomalies, n_anomalies / len(df) * 100
            )
            
            if n_anomalies > 0:
                logger.info("   Top 5 anomalies by feerate:")
                anomaly_df = df[df['is_anomaly']].nlargest(5, 'feerate')
                for _, row in anomaly_df.iterrows():
                    logger.info(
                        "     - %s... (feerate: %.2f, inputs: %d, outputs: %d)",
                        row['txid'][:16], row['feerate'],
                        row['inputs_count'], row['outputs_count']
                    )
        except FileNotFoundError:
            logger.warning("   Anomaly detector model not found")
            df['is_anomaly'] = None
        
        # Save results
        logger.info("\n5. Saving results...")
        output_cols = [
            'txid', 'block_height', 'feerate', 'predicted_feerate',
            'coinjoin_probability', 'cluster', 'is_anomaly'
        ]
        available_cols = [c for c in output_cols if c in df.columns]
        
        output_file = os.path.join(RESULTS_DIR, 'ml_predictions.csv')
        df[available_cols].to_csv(output_file, index=False)
        logger.info("   Results saved to: %s", output_file)
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info("Transactions analyzed: %d", len(df))
        
        if df['coinjoin_probability'].notna().any():
            logger.info("CoinJoin candidates: %d", (df['coinjoin_probability'] > 0.7).sum())
        if df['is_anomaly'].notna().any():
            logger.info("Anomalies: %d", df['is_anomaly'].sum())
        if df['cluster'].notna().any():
            logger.info("Clusters identified: %d", df['cluster'].nunique())
        
        return df
    
    def predict_optimal_feerate(
        self,
        inputs: int = 2,
        outputs: int = 2,
        vsize: int = 250
    ) -> Optional[float]:
        """
        Predict optimal fee rate for a hypothetical transaction.
        
        Args:
            inputs: Number of inputs.
            outputs: Number of outputs.
            vsize: Virtual size in vBytes.
            
        Returns:
            Predicted fee rate in sat/vB, or None on error.
        """
        logger.info("=" * 50)
        logger.info("Optimal Feerate Prediction")
        logger.info("=" * 50)
        
        try:
            now = datetime.now()
            
            # Create hypothetical transaction DataFrame
            df = pd.DataFrame([{
                'inputs_count': inputs,
                'outputs_count': outputs,
                'vsize': vsize,
                'size': int(vsize * 1.1),
                'weight': vsize * 4,
                'hour': now.hour,
                'day_of_week': now.weekday(),
                'is_weekend': 1 if now.weekday() >= 5 else 0,
                'is_rbf_int': 1,
                'io_ratio': inputs / outputs,
                'weight_to_size_ratio': 4.0,
            }])
            
            predicted_feerate = self.feerate_predictor.predict(df)[0]
            
            logger.info("\nTransaction parameters:")
            logger.info("  Inputs: %d", inputs)
            logger.info("  Outputs: %d", outputs)
            logger.info("  vSize: %d vB", vsize)
            logger.info("\nPredicted optimal feerate: %.2f sat/vB", predicted_feerate)
            logger.info("Estimated total fee: %d sat", int(predicted_feerate * vsize))
            
            return predicted_feerate
            
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return None


def main():
    """Entry point for predictions."""
    parser = argparse.ArgumentParser(
        description='ML predictions for Bitcoin transactions'
    )
    parser.add_argument(
        '--mode',
        choices=['analyze', 'predict_fee'],
        default='analyze',
        help='Execution mode'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Number of transactions to analyze'
    )
    parser.add_argument(
        '--inputs',
        type=int,
        default=2,
        help='Number of inputs (for predict_fee mode)'
    )
    parser.add_argument(
        '--outputs',
        type=int,
        default=2,
        help='Number of outputs (for predict_fee mode)'
    )
    parser.add_argument(
        '--vsize',
        type=int,
        default=250,
        help='Virtual size in vB (for predict_fee mode)'
    )
    
    args = parser.parse_args()
    
    analyzer = TransactionAnalyzer()
    
    if args.mode == 'analyze':
        analyzer.analyze(args.limit)
    elif args.mode == 'predict_fee':
        analyzer.predict_optimal_feerate(args.inputs, args.outputs, args.vsize)


if __name__ == "__main__":
    main()
