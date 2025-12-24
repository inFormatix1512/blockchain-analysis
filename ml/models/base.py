import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports if not present
# We need to go up 2 levels from ml/models/base.py to get to root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from common import Config

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
