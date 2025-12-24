import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import List
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from common import Config
from .base import BaseModel, MODEL_DIR

logger = logging.getLogger(__name__)

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
