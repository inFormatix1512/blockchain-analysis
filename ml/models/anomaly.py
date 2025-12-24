import logging
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import IsolationForest

from common import Config
from .base import BaseModel

logger = logging.getLogger(__name__)

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
