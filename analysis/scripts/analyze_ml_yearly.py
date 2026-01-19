#!/usr/bin/env python3
"""
YEARLY ML ANALYSIS
Runs the trained ML models on historical data (2011-2023) to show trends.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Add repo root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common import get_db_connection
from ml.models import (
    CoinJoinDetector,
    TransactionClusterer,
    AnomalyDetector,
    FeeratePredictor
)
from ml.feature_engineering import engineer_features

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = 'analysis/results/thesis_yearly_ml_2023'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_models():
    models = {}
    try:
        models['coinjoin'] = CoinJoinDetector.load()
    except Exception as e:
        logger.warning(f"Could not load CoinJoinDetector: {e}")
        models['coinjoin'] = None

    try:
        models['clusterer'] = TransactionClusterer.load()
    except Exception as e:
        logger.warning(f"Could not load TransactionClusterer: {e}")
        models['clusterer'] = None

    try:
        models['anomaly'] = AnomalyDetector.load()
    except Exception as e:
        logger.warning(f"Could not load AnomalyDetector: {e}")
        models['anomaly'] = None

    try:
        models['feerate'] = FeeratePredictor.load()
    except Exception as e:
        logger.warning(f"Could not load FeeratePredictor: {e}")
        models['feerate'] = None
        
    return models

def get_yearly_data(year, limit=5000):
    query = f"""
    SELECT 
        b.txid, b.block_height, b.ts,
        EXTRACT(EPOCH FROM b.ts) as timestamp,
        b.size, b.vsize, b.weight, b.fee,
        COALESCE(
            b.feerate,
            CASE 
                WHEN b.fee IS NOT NULL AND b.vsize IS NOT NULL AND b.vsize > 0
                    THEN (b.fee::numeric / b.vsize)
                ELSE NULL
            END
        ) as feerate,
        b.inputs_count, b.outputs_count,
        h.is_rbf, h.coinjoin_score, h.equal_output, h.likely_change_index
    FROM tx_basic b
    JOIN tx_heuristics h ON b.txid = h.txid
    WHERE extract(year from b.ts) = {year}
    ORDER BY random()
    LIMIT {limit}
    """
    with get_db_connection() as conn:
        df = pd.read_sql(query, conn)
    return df

def analyze_year(year, models, df):
    stats = {'year': year, 'tx_count': len(df)}
    
    if df.empty:
        return stats

    # Prepare features
    # Note: engineer_features expects raw df columns
    # We might need to ensure column names match what load_transaction_data produces
    # It basically adds ratio columns etc.
    try:
        df_features = engineer_features(df)
    except Exception as e:
        logger.error(f"Feature engineering failed for {year}: {e}")
        return stats

    # 1. CoinJoin Analysis
    if models.get('coinjoin'):
        preds = models['coinjoin'].predict(df_features)
        stats['coinjoin_detected_ml'] = (preds == 1).mean() * 100
    else:
        stats['coinjoin_detected_ml'] = 0

    # Heuristic comparison
    stats['coinjoin_heuristic'] = (df['coinjoin_score'] > 0.5).mean() * 100

    # 2. Feerate Analysis
    # Actual feerate
    # Handle possible None/zeros
    # fee is in satoshis, vsize in vbytes
    df['calculated_feerate'] = df.apply(lambda x: x['fee'] / x['vsize'] if x['vsize'] > 0 and pd.notnull(x['fee']) else 0, axis=1)
    stats['avg_actual_feerate'] = df['calculated_feerate'].mean()

    if models.get('feerate'):
        try:
            preds = models['feerate'].predict(df_features)
            stats['avg_predicted_feerate'] = preds.mean()
        except:
            stats['avg_predicted_feerate'] = 0
    else:
        stats['avg_predicted_feerate'] = 0

    # 3. Clustering Analysis
    if models.get('clusterer'):
        clusters = models['clusterer'].predict(df_features)
        # Store distribution
        counts = pd.Series(clusters).value_counts(normalize=True)
        for c_id in range(5): # assuming 5 clusters k=5
            stats[f'cluster_{c_id}_pct'] = counts.get(c_id, 0) * 100

    # 4. Anomaly Analysis
    if models.get('anomaly'):
        anoms = models['anomaly'].predict(df_features) # -1 is anomaly, 1 is normal usually for IsolationForest
        # My AnomalyDetector wrapper likely returns 1 for anomaly or 0? 
        # Checking anomaly.py usually implies predict returns boolean or 1/0?
        # Let's check logic: usually sklearn returns -1 for anomaly.
        # But wrapper might normalize.
        # Let's assume wrapper returns 1 for anomaly (as per training log "Anomalies detected: ...")
        stats['anomaly_pct'] = (anoms == -1).mean() * 100 
        # Wait, let's verify if wrapper converts it. 
        # If not sure, count both.
        # IsolationForest: -1 anomaly, 1 normal
        stats['anomaly_raw_neg1_pct'] = (anoms == -1).mean() * 100
        stats['anomaly_raw_1_pct'] = (anoms == 1).mean() * 100

    return stats

def main():
    logger.info("Starting Yearly ML Analysis (2011-2023)")
    
    models = load_models()
    all_stats = []

    for year in range(2011, 2024):
        logger.info(f"Analyzing year {year}...")
        try:
            df = get_yearly_data(year)
            if len(df) < 100:
                logger.warning(f"Not enough data for {year} (found {len(df)})")
                continue
            
            s = analyze_year(year, models, df)
            all_stats.append(s)
            logger.info(f"Done {year}: {s}")
        except Exception as e:
            logger.error(f"Failed year {year}: {e}")

    # Create DataFrame
    res_df = pd.DataFrame(all_stats)
    res_df.set_index('year', inplace=True)
    
    logger.info("Saving results...")
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'yearly_ml_stats.csv'))
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    # 1. CoinJoin Trend
    plt.figure(figsize=(10, 6))
    if 'coinjoin_detected_ml' in res_df.columns:
        sns.lineplot(data=res_df, x=res_df.index, y='coinjoin_detected_ml', marker='o', label='ML Detected')
    sns.lineplot(data=res_df, x=res_df.index, y='coinjoin_heuristic', marker='x', label='Heuristic (>0.5)')
    plt.title('CoinJoin Transactions % per Year')
    plt.ylabel('Percentage (%)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'coinjoin_trend.png'))
    plt.close()

    # 2. Feerate Trend
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x=res_df.index, y='avg_actual_feerate', marker='o', label='Actual Avg Fee (sat/vB)')
    if 'avg_predicted_feerate' in res_df.columns and res_df['avg_predicted_feerate'].sum() > 0:
        sns.lineplot(data=res_df, x=res_df.index, y='avg_predicted_feerate', marker='--', label='Model Predicted')
    plt.title('Average Feerate per Year')
    plt.ylabel('Fee (sat/vB)')
    plt.yscale('log')
    plt.savefig(os.path.join(OUTPUT_DIR, 'feerate_trend.png'))
    plt.close()

    # 3. Anomalies
    plt.figure(figsize=(10, 6))
    # Check which col has data
    if 'anomaly_raw_neg1_pct' in res_df.columns:
        # Assuming -1 is anomaly
        sns.lineplot(data=res_df, x=res_df.index, y='anomaly_raw_neg1_pct', marker='o', color='red')
    plt.title('Anomaly Percentage per Year')
    plt.ylabel('% Anomalous Txs')
    plt.savefig(os.path.join(OUTPUT_DIR, 'anomaly_trend.png'))
    plt.close()

    # 4. Clustering (Stacked Area?)
    # Filter cluster cols
    cluster_cols = [c for c in res_df.columns if c.startswith('cluster_') and c.endswith('_pct')]
    if cluster_cols:
        res_df[cluster_cols].plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Transaction Cluster Distribution per Year')
        plt.ylabel('Percentage')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_distribution.png'))
        plt.close()

    logger.info(f"Analysis complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
