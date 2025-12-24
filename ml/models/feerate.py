import logging
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from common import Config
from .base import BaseModel

logger = logging.getLogger(__name__)

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
