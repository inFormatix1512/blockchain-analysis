#!/usr/bin/env python3
"""
Modulo per training di modelli ML per analisi blockchain.
Implementa diversi modelli per classificazione e predizione.
"""

import os
import sys
import pandas as pd
import numpy as np
import psycopg2
import joblib
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    mean_squared_error, r2_score, silhouette_score
)
import warnings
warnings.filterwarnings('ignore')

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurazione DB
PGHOST = os.environ.get('PGHOST', 'localhost')
PGUSER = os.environ.get('PGUSER', 'postgres')
PGPASSWORD = os.environ.get('PGPASSWORD', 'postgres')
PGDATABASE = os.environ.get('PGDATABASE', 'blockchain')

# Directory per salvare modelli
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def get_db_connection():
    """Crea connessione al database."""
    return psycopg2.connect(
        host=PGHOST, 
        user=PGUSER, 
        password=PGPASSWORD, 
        dbname=PGDATABASE
    )


def load_transaction_data(min_samples=1000):
    """Carica dati transazioni dal database."""
    logger.info("Caricamento dati transazioni dal database...")
    
    conn = get_db_connection()
    
    query = """
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
    LIMIT 100000
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    logger.info(f"Caricati {len(df)} record")
    
    if len(df) < min_samples:
        logger.warning(f"Dati insufficienti: {len(df)} < {min_samples}. Alcuni modelli potrebbero non funzionare.")
    
    return df


def engineer_features(df):
    """Feature engineering per ML."""
    logger.info("Engineering features...")
    
    # Features derivate
    df['fee_per_input'] = df['fee'] / df['inputs_count']
    df['fee_per_output'] = df['fee'] / df['outputs_count']
    df['io_ratio'] = df['inputs_count'] / df['outputs_count']
    df['size_per_input'] = df['vsize'] / df['inputs_count']
    df['size_per_output'] = df['vsize'] / df['outputs_count']
    df['weight_to_size_ratio'] = df['weight'] / df['size']
    
    # Features temporali
    df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Features categoriche
    df['is_rbf_int'] = df['is_rbf'].astype(int)
    df['equal_output_int'] = df['equal_output'].astype(int)
    df['has_change'] = df['likely_change_index'].notna().astype(int)
    
    # Rimuovi infiniti e NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df


class CoinJoinDetector:
    """Modello per rilevare transazioni coinjoin."""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, df):
        """Prepara dati per training."""
        # Label: coinjoin se equal_output=True e coinjoin_score > 0.5
        df['is_coinjoin'] = ((df['equal_output'] == True) & 
                             (df['coinjoin_score'] > 0.5)).astype(int)
        
        # Features
        feature_cols = [
            'vsize', 'inputs_count', 'outputs_count', 
            'feerate', 'io_ratio', 'coinjoin_score',
            'fee_per_input', 'fee_per_output', 'weight_to_size_ratio'
        ]
        
        X = df[feature_cols].values
        y = df['is_coinjoin'].values
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, df):
        """Addestra il modello."""
        logger.info("Training CoinJoin Detector...")
        
        X, y = self.prepare_data(df)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Training
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        logger.info("\nCoinJoin Detector Performance:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nFeature Importances:")
        logger.info(f"\n{importances}")
        
        return self
    
    def predict(self, df):
        """Predice probabilità coinjoin."""
        X, _ = self.prepare_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, filename='coinjoin_detector.pkl'):
        """Salva il modello."""
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Modello salvato in {path}")
    
    @classmethod
    def load(cls, filename='coinjoin_detector.pkl'):
        """Carica il modello."""
        path = os.path.join(MODEL_DIR, filename)
        data = joblib.load(path)
        detector = cls()
        detector.model = data['model']
        detector.scaler = data['scaler']
        detector.feature_names = data['feature_names']
        return detector


class FeeratePredictor:
    """Modello per predire feerate ottimale."""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, df):
        """Prepara dati per training."""
        feature_cols = [
            'inputs_count', 'outputs_count', 'vsize',
            'hour', 'day_of_week', 'is_weekend',
            'is_rbf_int', 'io_ratio', 'weight_to_size_ratio'
        ]
        
        X = df[feature_cols].values
        y = np.log1p(df['feerate'].values)  # Log transform per normalizzare
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, df):
        """Addestra il modello."""
        logger.info("Training Feerate Predictor...")
        
        X, y = self.prepare_data(df)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Training
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info("\nFeerate Predictor Performance:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"RMSE: {np.sqrt(mse):.4f}")
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nFeature Importances:")
        logger.info(f"\n{importances}")
        
        return self
    
    def predict(self, df):
        """Predice feerate."""
        X, _ = self.prepare_data(df)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return np.expm1(y_pred)  # Inverse log transform
    
    def save(self, filename='feerate_predictor.pkl'):
        """Salva il modello."""
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Modello salvato in {path}")
    
    @classmethod
    def load(cls, filename='feerate_predictor.pkl'):
        """Carica il modello."""
        path = os.path.join(MODEL_DIR, filename)
        data = joblib.load(path)
        predictor = cls()
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_names = data['feature_names']
        return predictor


class TransactionClusterer:
    """Clustering di transazioni per pattern analysis."""
    
    def __init__(self, method='kmeans', n_clusters=5):
        self.method = method
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.feature_names = None
        
        if method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Metodo {method} non supportato")
    
    def prepare_data(self, df):
        """Prepara dati per clustering."""
        feature_cols = [
            'vsize', 'inputs_count', 'outputs_count',
            'feerate', 'coinjoin_score', 'io_ratio',
            'fee_per_input', 'weight_to_size_ratio'
        ]
        
        X = df[feature_cols].values
        self.feature_names = feature_cols
        
        return X
    
    def fit(self, df):
        """Esegue clustering."""
        logger.info(f"Clustering con {self.method}...")
        
        X = self.prepare_data(df)
        X_scaled = self.scaler.fit_transform(X)
        
        # Clustering
        labels = self.model.fit_predict(X_scaled)
        
        # Evaluation
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            logger.info(f"Silhouette Score: {score:.4f}")
        
        # Analizza clusters
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        
        logger.info("\nCluster Statistics:")
        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            logger.info(f"\nCluster {cluster_id} ({len(cluster_data)} transactions):")
            logger.info(f"  Avg feerate: {cluster_data['feerate'].mean():.2f}")
            logger.info(f"  Avg inputs: {cluster_data['inputs_count'].mean():.2f}")
            logger.info(f"  Avg outputs: {cluster_data['outputs_count'].mean():.2f}")
            logger.info(f"  RBF rate: {cluster_data['is_rbf'].mean():.2%}")
            logger.info(f"  CoinJoin rate: {cluster_data['equal_output'].mean():.2%}")
        
        return labels
    
    def save(self, filename='transaction_clusterer.pkl'):
        """Salva il modello."""
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'method': self.method,
            'n_clusters': self.n_clusters
        }, path)
        logger.info(f"Modello salvato in {path}")
    
    @classmethod
    def load(cls, filename='transaction_clusterer.pkl'):
        """Carica il modello."""
        path = os.path.join(MODEL_DIR, filename)
        data = joblib.load(path)
        clusterer = cls(method=data['method'], n_clusters=data['n_clusters'])
        clusterer.model = data['model']
        clusterer.scaler = data['scaler']
        clusterer.feature_names = data['feature_names']
        return clusterer


class AnomalyDetector:
    """Rilevamento anomalie nelle transazioni."""
    
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, df):
        """Prepara dati per anomaly detection."""
        feature_cols = [
            'vsize', 'inputs_count', 'outputs_count',
            'feerate', 'fee', 'io_ratio',
            'fee_per_input', 'fee_per_output'
        ]
        
        X = df[feature_cols].values
        self.feature_names = feature_cols
        
        return X
    
    def fit(self, df):
        """Addestra il detector."""
        logger.info("Training Anomaly Detector...")
        
        X = self.prepare_data(df)
        X_scaled = self.scaler.fit_transform(X)
        
        # Training
        predictions = self.model.fit_predict(X_scaled)
        
        # -1 per anomalie, 1 per normali
        n_anomalies = (predictions == -1).sum()
        logger.info(f"Anomalie rilevate: {n_anomalies} ({n_anomalies/len(df):.2%})")
        
        return predictions
    
    def predict(self, df):
        """Predice anomalie."""
        X = self.prepare_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filename='anomaly_detector.pkl'):
        """Salva il modello."""
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Modello salvato in {path}")
    
    @classmethod
    def load(cls, filename='anomaly_detector.pkl'):
        """Carica il modello."""
        path = os.path.join(MODEL_DIR, filename)
        data = joblib.load(path)
        detector = cls()
        detector.model = data['model']
        detector.scaler = data['scaler']
        detector.feature_names = data['feature_names']
        return detector


def main():
    """Training principale di tutti i modelli."""
    logger.info("=== Inizio training modelli ML ===")
    
    try:
        # Carica dati
        df = load_transaction_data()
        
        if len(df) < 100:
            logger.error("Dati insufficienti per training. Raccogli più dati prima.")
            return
        
        # Feature engineering
        df = engineer_features(df)
        
        # 1. CoinJoin Detector
        logger.info("\n" + "="*50)
        coinjoin_detector = CoinJoinDetector()
        coinjoin_detector.train(df)
        coinjoin_detector.save()
        
        # 2. Feerate Predictor
        logger.info("\n" + "="*50)
        feerate_predictor = FeeratePredictor()
        feerate_predictor.train(df)
        feerate_predictor.save()
        
        # 3. Transaction Clusterer
        logger.info("\n" + "="*50)
        clusterer = TransactionClusterer(method='kmeans', n_clusters=5)
        clusterer.fit(df)
        clusterer.save()
        
        # 4. Anomaly Detector
        logger.info("\n" + "="*50)
        anomaly_detector = AnomalyDetector(contamination=0.05)
        anomaly_detector.fit(df)
        anomaly_detector.save()
        
        logger.info("\n=== Training completato con successo! ===")
        logger.info(f"Modelli salvati in: {MODEL_DIR}")
        
    except Exception as e:
        logger.error(f"Errore durante training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
