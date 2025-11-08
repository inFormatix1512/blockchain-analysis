#!/usr/bin/env python3
"""
Script per fare predizioni usando i modelli ML addestrati.
"""

import os
import sys
import pandas as pd
import psycopg2
import logging
from train_models import (
    CoinJoinDetector, FeeratePredictor, 
    TransactionClusterer, AnomalyDetector,
    get_db_connection, engineer_features
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


def load_recent_transactions(limit=1000):
    """Carica transazioni recenti per analisi."""
    logger.info(f"Caricamento ultime {limit} transazioni...")
    
    conn = get_db_connection()
    
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
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    logger.info(f"Caricate {len(df)} transazioni")
    
    return df


def analyze_transactions():
    """Analizza transazioni con tutti i modelli."""
    logger.info("=== Inizio analisi ML ===\n")
    
    # Carica dati
    df = load_recent_transactions(limit=1000)
    
    if df.empty:
        logger.error("Nessuna transazione trovata")
        return
    
    # Feature engineering
    df = engineer_features(df)
    
    try:
        # 1. CoinJoin Detection
        logger.info("1. Analisi CoinJoin...")
        coinjoin_detector = CoinJoinDetector.load()
        coinjoin_probs = coinjoin_detector.predict(df)
        df['coinjoin_probability'] = coinjoin_probs
        
        high_coinjoin = df[df['coinjoin_probability'] > 0.7]
        logger.info(f"   Transazioni con alta probabilità CoinJoin: {len(high_coinjoin)}")
        if not high_coinjoin.empty:
            logger.info(f"   Top 5 CoinJoin candidates:")
            for idx, row in high_coinjoin.nlargest(5, 'coinjoin_probability').iterrows():
                logger.info(f"     - {row['txid'][:16]}... (prob: {row['coinjoin_probability']:.2%})")
        
        # 2. Feerate Prediction
        logger.info("\n2. Predizione Feerate...")
        feerate_predictor = FeeratePredictor.load()
        predicted_feerates = feerate_predictor.predict(df)
        df['predicted_feerate'] = predicted_feerates
        df['feerate_diff'] = df['feerate'] - df['predicted_feerate']
        
        logger.info(f"   Feerate medio attuale: {df['feerate'].mean():.2f} sat/vB")
        logger.info(f"   Feerate medio predetto: {df['predicted_feerate'].mean():.2f} sat/vB")
        
        # Transazioni che pagano troppo o troppo poco
        overpaying = df[df['feerate_diff'] > df['feerate'].std()]
        underpaying = df[df['feerate_diff'] < -df['feerate'].std()]
        
        logger.info(f"   Transazioni che pagano troppo: {len(overpaying)}")
        logger.info(f"   Transazioni che pagano troppo poco: {len(underpaying)}")
        
        # 3. Clustering
        logger.info("\n3. Clustering transazioni...")
        clusterer = TransactionClusterer.load()
        X = clusterer.prepare_data(df)
        X_scaled = clusterer.scaler.transform(X)
        clusters = clusterer.model.predict(X_scaled)
        df['cluster'] = clusters
        
        logger.info(f"   Distribuzione clusters:")
        for cluster_id in sorted(df['cluster'].unique()):
            count = (df['cluster'] == cluster_id).sum()
            logger.info(f"     Cluster {cluster_id}: {count} transazioni ({count/len(df):.1%})")
        
        # 4. Anomaly Detection
        logger.info("\n4. Rilevamento anomalie...")
        anomaly_detector = AnomalyDetector.load()
        anomalies = anomaly_detector.predict(df)
        df['is_anomaly'] = anomalies == -1
        
        n_anomalies = df['is_anomaly'].sum()
        logger.info(f"   Anomalie rilevate: {n_anomalies} ({n_anomalies/len(df):.1%})")
        
        if n_anomalies > 0:
            logger.info(f"   Top 5 anomalie:")
            anomaly_df = df[df['is_anomaly']].nlargest(5, 'feerate')
            for idx, row in anomaly_df.iterrows():
                logger.info(f"     - {row['txid'][:16]}... (feerate: {row['feerate']:.2f}, "
                          f"inputs: {row['inputs_count']}, outputs: {row['outputs_count']})")
        
        # Salva risultati
        logger.info("\n5. Salvataggio risultati...")
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, 'ml_predictions.csv')
        df[['txid', 'block_height', 'feerate', 'predicted_feerate', 
            'coinjoin_probability', 'cluster', 'is_anomaly']].to_csv(output_file, index=False)
        
        logger.info(f"   Risultati salvati in: {output_file}")
        
        # Summary
        logger.info("\n=== Summary ===")
        logger.info(f"Transazioni analizzate: {len(df)}")
        logger.info(f"CoinJoin candidates: {(df['coinjoin_probability'] > 0.7).sum()}")
        logger.info(f"Anomalie: {n_anomalies}")
        logger.info(f"Clusters identificati: {df['cluster'].nunique()}")
        
    except FileNotFoundError as e:
        logger.error(f"Modello non trovato: {e}")
        logger.error("Esegui prima 'python train_models.py' per addestrare i modelli")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Errore durante analisi: {e}", exc_info=True)
        sys.exit(1)


def predict_optimal_feerate(inputs=2, outputs=2, vsize=250):
    """Predice feerate ottimale per una transazione."""
    logger.info("=== Predizione Feerate Ottimale ===")
    
    try:
        feerate_predictor = FeeratePredictor.load()
        
        # Crea dataframe con transazione ipotetica
        import datetime
        now = datetime.datetime.now()
        
        df = pd.DataFrame([{
            'inputs_count': inputs,
            'outputs_count': outputs,
            'vsize': vsize,
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'is_weekend': 1 if now.weekday() >= 5 else 0,
            'is_rbf_int': 1,  # Assumiamo RBF abilitato
            'io_ratio': inputs / outputs,
            'weight_to_size_ratio': 4.0,  # Tipico per SegWit
            'size': vsize * 1.1,
            'weight': vsize * 4
        }])
        
        predicted_feerate = feerate_predictor.predict(df)[0]
        
        logger.info(f"\nParametri transazione:")
        logger.info(f"  Inputs: {inputs}")
        logger.info(f"  Outputs: {outputs}")
        logger.info(f"  vSize: {vsize} vB")
        logger.info(f"\nFeerate ottimale predetto: {predicted_feerate:.2f} sat/vB")
        logger.info(f"Fee totale stimata: {predicted_feerate * vsize:.0f} sat")
        
        return predicted_feerate
        
    except Exception as e:
        logger.error(f"Errore: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML predictions per transazioni Bitcoin')
    parser.add_argument('--mode', choices=['analyze', 'predict_fee'], 
                       default='analyze', help='Modalità di esecuzione')
    parser.add_argument('--inputs', type=int, default=2, help='Numero di input')
    parser.add_argument('--outputs', type=int, default=2, help='Numero di output')
    parser.add_argument('--vsize', type=int, default=250, help='Virtual size in vB')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        analyze_transactions()
    elif args.mode == 'predict_fee':
        predict_optimal_feerate(args.inputs, args.outputs, args.vsize)
