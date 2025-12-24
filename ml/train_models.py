#!/usr/bin/env python3
"""
Machine Learning models for blockchain transaction analysis.
Orchestrator script for training all models.
"""

import logging
import sys
import warnings
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from common import Config
from ml.models import (
    BaseModel,
    CoinJoinDetector,
    FeeratePredictor,
    TransactionClusterer,
    AnomalyDetector
)
from ml.feature_engineering import load_transaction_data, engineer_features

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_DIR = Config().ml.model_dir

def train_all_models() -> Dict[str, BaseModel]:
    """
    Train all ML models.
    
    Returns:
        Dictionary of trained model instances.
    """
    logger.info("=" * 50)
    logger.info("Starting ML model training")
    logger.info("=" * 50)
    
    # Load and prepare data
    logger.info("Loading data (limit=20000 for faster demo)...")
    df = load_transaction_data(limit=20000)
    
    if len(df) < 100:
        logger.error("Insufficient data for training. Collect more data first.")
        return {}
    
    df = engineer_features(df)
    
    models = {}
    
    # 1. CoinJoin Detector
    logger.info("\n" + "=" * 50)
    coinjoin_detector = CoinJoinDetector()
    coinjoin_detector.train(df)
    if getattr(coinjoin_detector, '_is_fitted', False):
        coinjoin_detector.save()
        models['coinjoin_detector'] = coinjoin_detector
    else:
        logger.warning("CoinJoin detector not trained (insufficient class variety).")
    
    # 2. Feerate Predictor (optional: requires enough non-zero feerate labels)
    logger.info("\n" + "=" * 50)
    n_positive_feerate = int((df.get('feerate', 0) > 0).sum()) if 'feerate' in df.columns else 0
    if n_positive_feerate >= 1000:
        feerate_predictor = FeeratePredictor()
        feerate_predictor.train(df)
        if getattr(feerate_predictor, '_is_fitted', False):
            feerate_predictor.save()
            models['feerate_predictor'] = feerate_predictor
        else:
            logger.warning("Feerate predictor training skipped (model not fitted).")
    else:
        logger.warning(
            "Skipping Feerate Predictor training: not enough feerate>0 samples (%d).",
            n_positive_feerate
        )
    
    # 3. Transaction Clusterer
    logger.info("\n" + "=" * 50)
    clusterer = TransactionClusterer(
        method='kmeans',
        n_clusters=Config().ml.n_clusters
    )
    clusterer.fit(df)
    if getattr(clusterer, '_is_fitted', False):
        clusterer.save()
        models['clusterer'] = clusterer
    else:
        logger.warning("Clusterer not fitted; skipping save.")
    
    # 4. Anomaly Detector
    logger.info("\n" + "=" * 50)
    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit(df)
    if getattr(anomaly_detector, '_is_fitted', False):
        anomaly_detector.save()
        models['anomaly_detector'] = anomaly_detector
    else:
        logger.warning("Anomaly detector not fitted; skipping save.")
    
    logger.info("\n" + "=" * 50)
    logger.info("Training completed successfully!")
    logger.info("Models saved to: %s", MODEL_DIR)
    
    return models


def main():
    """Entry point for model training."""
    try:
        train_all_models()
    except Exception as e:
        logger.error("Error during training: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
