import logging
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from common import Config
from .base import BaseModel

logger = logging.getLogger(__name__)

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
            'feerate', 'io_ratio',
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
        
        # Check if we have enough positive samples
        if len(np.unique(y)) < 2:
            logger.warning("Skipping CoinJoin Detector training: Only one class present in data (CoinJoins found: %d)", y.sum())
            return self

        # Stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=Config().ml.test_size,
                random_state=Config().ml.random_state,
                stratify=y
            )
        except ValueError as e:
            logger.warning("Skipping CoinJoin Detector training: Not enough samples for stratified split. %s", e)
            return self
        
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
