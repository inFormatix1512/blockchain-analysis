#!/usr/bin/env python3
"""
Machine Learning models for blockchain transaction analysis.

This module implements various ML models for:
- CoinJoin transaction detection (classification)
- Fee rate prediction (regression)
- Transaction clustering (unsupervised)
- Anomaly detection (isolation forest)
"""

import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from common import Config, get_db_connection

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model directory from configuration
MODEL_DIR = Config().ml.model_dir
os.makedirs(MODEL_DIR, exist_ok=True)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    r2: Optional[float] = None
    silhouette: Optional[float] = None


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    Provides common functionality for model training, prediction,
    serialization, and feature management.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    @abstractmethod
    def _get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        pass
    
    @abstractmethod
    def _prepare_target(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare target variable (if applicable)."""
        pass
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from DataFrame.
        
        Args:
            df: DataFrame with engineered features.
            
        Returns:
            Feature matrix as numpy array.
        """
        self.feature_names = self._get_feature_columns()
        return df[self.feature_names].values
    
    def save(self, filename: str) -> str:
        """
        Save model to disk.
        
        Args:
            filename: Model filename.
            
        Returns:
            Full path to saved model.
        """
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self._is_fitted,
        }, path)
        logger.info("Model saved to %s", path)
        return path
    
    @classmethod
    def load(cls, filename: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            filename: Model filename.
            
        Returns:
            Loaded model instance.
        """
        path = os.path.join(MODEL_DIR, filename)
        data = joblib.load(path)
        
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance._is_fitted = data.get('is_fitted', True)
        
        return instance
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.
        
        Returns:
            DataFrame with feature importances or None.
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


class CoinJoinDetector(BaseModel):
    """
    Binary classifier for CoinJoin transaction detection.
    
    Uses Random Forest to identify transactions that exhibit
    CoinJoin characteristics based on structural features.
    """
    
    DEFAULT_FILENAME = 'coinjoin_detector.pkl'
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            random_state=Config().ml.random_state,
            n_jobs=-1
        )
    
    def _get_feature_columns(self) -> List[str]:
        return [
            'vsize', 'inputs_count', 'outputs_count',
            'feerate', 'io_ratio', 'coinjoin_score',
            'fee_per_input', 'fee_per_output', 'weight_to_size_ratio'
        ]
    
    def _prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create binary CoinJoin label."""
        is_coinjoin = (
            (df['equal_output'] == True) & 
            (df['coinjoin_score'] > 0.5)
        ).astype(int)
        return is_coinjoin.values
    
    def train(self, df: pd.DataFrame) -> 'CoinJoinDetector':
        """
        Train the CoinJoin detector.
        
        Args:
            df: DataFrame with engineered features.
            
        Returns:
            Self for method chaining.
        """
        logger.info("Training CoinJoin Detector...")
        
        X = self.prepare_features(df)
        y = self._prepare_target(df)
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config().ml.test_size,
            random_state=Config().ml.random_state,
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        self._is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        logger.info("\nCoinJoin Detector Performance:")
        logger.info("\n%s", classification_report(y_test, y_pred))
        
        # Log feature importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            logger.info("\nFeature Importances:\n%s", importance_df)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict CoinJoin probability.
        
        Args:
            df: DataFrame with features.
            
        Returns:
            Array of CoinJoin probabilities.
        """
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, filename: str = DEFAULT_FILENAME) -> str:
        return super().save(filename)
    
    @classmethod
    def load(cls, filename: str = DEFAULT_FILENAME) -> 'CoinJoinDetector':
        return super().load.__func__(cls, filename)


class FeeratePredictor(BaseModel):
    """
    Regression model for optimal fee rate prediction.
    
    Uses Gradient Boosting to predict appropriate fee rates
    based on transaction characteristics and temporal features.
    """
    
    DEFAULT_FILENAME = 'feerate_predictor.pkl'
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=Config().ml.random_state
        )
    
    def _get_feature_columns(self) -> List[str]:
        return [
            'inputs_count', 'outputs_count', 'vsize',
            'hour', 'day_of_week', 'is_weekend',
            'is_rbf_int', 'io_ratio', 'weight_to_size_ratio'
        ]
    
    def _prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Log-transform feerate for better distribution."""
        return np.log1p(df['feerate'].values)
    
    def train(self, df: pd.DataFrame) -> 'FeeratePredictor':
        """
        Train the fee rate predictor.
        
        Args:
            df: DataFrame with engineered features.
            
        Returns:
            Self for method chaining.
        """
        logger.info("Training Feerate Predictor...")
        
        X = self.prepare_features(df)
        y = self._prepare_target(df)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config().ml.test_size,
            random_state=Config().ml.random_state
        )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        self._is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info("\nFeerate Predictor Performance:")
        logger.info("MSE: %.4f", mse)
        logger.info("R2 Score: %.4f", r2)
        logger.info("RMSE: %.4f", np.sqrt(mse))
        
        # Log feature importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            logger.info("\nFeature Importances:\n%s", importance_df)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict optimal fee rates.
        
        Args:
            df: DataFrame with features.
            
        Returns:
            Array of predicted fee rates (sat/vB).
        """
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return np.expm1(y_pred)  # Inverse log transform
    
    def save(self, filename: str = DEFAULT_FILENAME) -> str:
        return super().save(filename)
    
    @classmethod
    def load(cls, filename: str = DEFAULT_FILENAME) -> 'FeeratePredictor':
        return super().load.__func__(cls, filename)


class TransactionClusterer(BaseModel):
    """
    Unsupervised clustering for transaction pattern analysis.
    
    Groups transactions into clusters based on structural
    similarity to identify behavioral patterns.
    """
    
    DEFAULT_FILENAME = 'transaction_clusterer.pkl'
    
    def __init__(self, method: str = 'kmeans', n_clusters: int = 5):
        super().__init__()
        self.method = method
        self.n_clusters = n_clusters
        
        if method == 'kmeans':
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=Config().ml.random_state,
                n_init=10
            )
        elif method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    
    def _get_feature_columns(self) -> List[str]:
        return [
            'vsize', 'inputs_count', 'outputs_count',
            'feerate', 'coinjoin_score', 'io_ratio',
            'fee_per_input', 'weight_to_size_ratio'
        ]
    
    def _prepare_target(self, df: pd.DataFrame) -> None:
        """No target for unsupervised learning."""
        return None
    
    def fit(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit clustering model and return labels.
        
        Args:
            df: DataFrame with engineered features.
            
        Returns:
            Array of cluster labels.
        """
        logger.info("Clustering with %s...", self.method)
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        labels = self.model.fit_predict(X_scaled)
        self._is_fitted = True
        
        # Evaluate
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            score = silhouette_score(X_scaled, labels)
            logger.info("Silhouette Score: %.4f", score)
        
        # Log cluster statistics
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        
        logger.info("\nCluster Statistics:")
        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            logger.info(
                "\nCluster %d (%d transactions):",
                cluster_id, len(cluster_data)
            )
            logger.info("  Avg feerate: %.2f", cluster_data['feerate'].mean())
            logger.info("  Avg inputs: %.2f", cluster_data['inputs_count'].mean())
            logger.info("  Avg outputs: %.2f", cluster_data['outputs_count'].mean())
            logger.info("  RBF rate: %.2%%", cluster_data['is_rbf'].mean() * 100)
            logger.info("  CoinJoin rate: %.2%%", cluster_data['equal_output'].mean() * 100)
        
        return labels
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            df: DataFrame with features.
            
        Returns:
            Array of cluster labels.
        """
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filename: str = DEFAULT_FILENAME) -> str:
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'method': self.method,
            'n_clusters': self.n_clusters,
            'is_fitted': self._is_fitted,
        }, path)
        logger.info("Model saved to %s", path)
        return path
    
    @classmethod
    def load(cls, filename: str = DEFAULT_FILENAME) -> 'TransactionClusterer':
        path = os.path.join(MODEL_DIR, filename)
        data = joblib.load(path)
        
        instance = cls(method=data['method'], n_clusters=data['n_clusters'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance._is_fitted = data.get('is_fitted', True)
        
        return instance


class AnomalyDetector(BaseModel):
    """
    Anomaly detection using Isolation Forest.
    
    Identifies unusual transactions that deviate significantly
    from normal patterns in the dataset.
    """
    
    DEFAULT_FILENAME = 'anomaly_detector.pkl'
    
    def __init__(self, contamination: float = None):
        super().__init__()
        contamination = contamination or Config().ml.contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=Config().ml.random_state,
            n_jobs=-1
        )
    
    def _get_feature_columns(self) -> List[str]:
        return [
            'vsize', 'inputs_count', 'outputs_count',
            'feerate', 'fee', 'io_ratio',
            'fee_per_input', 'fee_per_output'
        ]
    
    def _prepare_target(self, df: pd.DataFrame) -> None:
        """No target for anomaly detection."""
        return None
    
    def fit(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit anomaly detector and return predictions.
        
        Args:
            df: DataFrame with engineered features.
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal).
        """
        logger.info("Training Anomaly Detector...")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        predictions = self.model.fit_predict(X_scaled)
        self._is_fitted = True
        
        n_anomalies = (predictions == -1).sum()
        logger.info(
            "Anomalies detected: %d (%.2f%%)",
            n_anomalies, n_anomalies / len(df) * 100
        )
        
        return predictions
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies for new data.
        
        Args:
            df: DataFrame with features.
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal).
        """
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filename: str = DEFAULT_FILENAME) -> str:
        return super().save(filename)
    
    @classmethod
    def load(cls, filename: str = DEFAULT_FILENAME) -> 'AnomalyDetector':
        return super().load.__func__(cls, filename)


# Data loading and feature engineering utilities

def load_transaction_data(min_samples: int = 1000, limit: int = 100000) -> pd.DataFrame:
    """
    Load transaction data from database.
    
    Args:
        min_samples: Minimum samples required for training.
        limit: Maximum number of records to load.
        
    Returns:
        DataFrame with transaction data.
    """
    logger.info("Loading transaction data from database...")
    
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
    
    logger.info("Loaded %d records", len(df))
    
    if len(df) < min_samples:
        logger.warning(
            "Insufficient data: %d < %d. Some models may not work properly.",
            len(df), min_samples
        )
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for ML models.
    
    Args:
        df: Raw transaction DataFrame.
        
    Returns:
        DataFrame with engineered features.
    """
    logger.info("Engineering features...")
    
    df = df.copy()
    
    # Ratio features
    df['fee_per_input'] = df['fee'] / df['inputs_count']
    df['fee_per_output'] = df['fee'] / df['outputs_count']
    df['io_ratio'] = df['inputs_count'] / df['outputs_count']
    df['size_per_input'] = df['vsize'] / df['inputs_count']
    df['size_per_output'] = df['vsize'] / df['outputs_count']
    df['weight_to_size_ratio'] = df['weight'] / df['size']
    
    # Temporal features
    timestamps = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = timestamps.dt.hour
    df['day_of_week'] = timestamps.dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Binary features
    df['is_rbf_int'] = df['is_rbf'].astype(int)
    df['equal_output_int'] = df['equal_output'].astype(int)
    df['has_change'] = df['likely_change_index'].notna().astype(int)
    
    # Clean up invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df


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
    df = load_transaction_data()
    
    if len(df) < 100:
        logger.error("Insufficient data for training. Collect more data first.")
        return {}
    
    df = engineer_features(df)
    
    models = {}
    
    # 1. CoinJoin Detector
    logger.info("\n" + "=" * 50)
    coinjoin_detector = CoinJoinDetector()
    coinjoin_detector.train(df)
    coinjoin_detector.save()
    models['coinjoin_detector'] = coinjoin_detector
    
    # 2. Feerate Predictor
    logger.info("\n" + "=" * 50)
    feerate_predictor = FeeratePredictor()
    feerate_predictor.train(df)
    feerate_predictor.save()
    models['feerate_predictor'] = feerate_predictor
    
    # 3. Transaction Clusterer
    logger.info("\n" + "=" * 50)
    clusterer = TransactionClusterer(
        method='kmeans',
        n_clusters=Config().ml.n_clusters
    )
    clusterer.fit(df)
    clusterer.save()
    models['clusterer'] = clusterer
    
    # 4. Anomaly Detector
    logger.info("\n" + "=" * 50)
    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit(df)
    anomaly_detector.save()
    models['anomaly_detector'] = anomaly_detector
    
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
