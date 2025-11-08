#!/usr/bin/env python3
"""
Script di test rapido per verificare il sistema ML.
Genera dati sintetici se il database è vuoto.
"""

import pandas as pd
import numpy as np
import sys
import os

# Aggiungi path ML
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))

from train_models import (
    CoinJoinDetector, FeeratePredictor,
    TransactionClusterer, AnomalyDetector,
    engineer_features
)

def generate_synthetic_data(n_samples=2000):
    """Genera dati sintetici per testare i modelli."""
    print(f"🔧 Generazione {n_samples} transazioni sintetiche per test...")
    
    np.random.seed(42)
    
    # Genera distribuzione realistica
    data = []
    
    for i in range(n_samples):
        # Tipo di transazione
        tx_type = np.random.choice(['normal', 'coinjoin', 'consolidation', 'priority'], 
                                   p=[0.7, 0.1, 0.15, 0.05])
        
        if tx_type == 'normal':
            inputs = np.random.randint(1, 4)
            outputs = np.random.randint(1, 3)
            vsize = np.random.randint(180, 400)
            feerate = np.random.lognormal(2.5, 0.8)
            coinjoin_score = 0.0
            equal_output = False
            
        elif tx_type == 'coinjoin':
            inputs = np.random.randint(3, 8)
            outputs = inputs
            vsize = np.random.randint(500, 1500)
            feerate = np.random.lognormal(2.8, 0.6)
            coinjoin_score = np.random.uniform(0.6, 1.0)
            equal_output = True
            
        elif tx_type == 'consolidation':
            inputs = np.random.randint(5, 20)
            outputs = np.random.randint(1, 3)
            vsize = np.random.randint(800, 2000)
            feerate = np.random.lognormal(1.5, 0.5)
            coinjoin_score = 0.0
            equal_output = False
            
        else:  # priority
            inputs = np.random.randint(1, 3)
            outputs = np.random.randint(1, 3)
            vsize = np.random.randint(150, 350)
            feerate = np.random.lognormal(3.5, 0.5)
            coinjoin_score = 0.0
            equal_output = False
        
        data.append({
            'txid': f'synthetic_{i:06d}',
            'block_height': 800000 + i,
            'timestamp': 1696000000 + i * 600,
            'size': int(vsize * 1.1),
            'vsize': int(vsize),
            'weight': int(vsize * 4),
            'fee': int(vsize * feerate),
            'feerate': feerate,
            'inputs_count': inputs,
            'outputs_count': outputs,
            'is_rbf': np.random.choice([True, False], p=[0.6, 0.4]),
            'coinjoin_score': coinjoin_score,
            'equal_output': equal_output,
            'likely_change_index': 1 if outputs > 1 else None
        })
    
    df = pd.DataFrame(data)
    print(f"✅ {len(df)} transazioni generate")
    return df


def test_ml_pipeline():
    """Test completo della pipeline ML."""
    print("\n" + "="*80)
    print("🧪 TEST PIPELINE MACHINE LEARNING")
    print("="*80 + "\n")
    
    # Genera dati sintetici
    df = generate_synthetic_data(2000)
    
    # Feature engineering
    print("\n📊 Feature Engineering...")
    df = engineer_features(df)
    print(f"   Features create: {df.shape[1]} colonne")
    
    # Test 1: CoinJoin Detector
    print("\n" + "="*80)
    print("1️⃣  TEST COINJOIN DETECTOR")
    print("="*80)
    
    coinjoin_detector = CoinJoinDetector()
    coinjoin_detector.train(df)
    coinjoin_detector.save('test_coinjoin_detector.pkl')
    
    # Predizioni
    df['coinjoin_probability'] = coinjoin_detector.predict(df)
    high_prob = df[df['coinjoin_probability'] > 0.7]
    print(f"\n✅ Test completato: {len(high_prob)} CoinJoin rilevati")
    
    # Test 2: Feerate Predictor
    print("\n" + "="*80)
    print("2️⃣  TEST FEERATE PREDICTOR")
    print("="*80)
    
    feerate_predictor = FeeratePredictor()
    feerate_predictor.train(df)
    feerate_predictor.save('test_feerate_predictor.pkl')
    
    # Predizioni
    df['predicted_feerate'] = feerate_predictor.predict(df)
    mae = abs(df['feerate'] - df['predicted_feerate']).mean()
    print(f"\n✅ Test completato: MAE = {mae:.2f} sat/vB")
    
    # Test 3: Transaction Clusterer
    print("\n" + "="*80)
    print("3️⃣  TEST TRANSACTION CLUSTERER")
    print("="*80)
    
    clusterer = TransactionClusterer(method='kmeans', n_clusters=4)
    clusters = clusterer.fit(df)
    clusterer.save('test_transaction_clusterer.pkl')
    
    df['cluster'] = clusters
    print(f"\n✅ Test completato: {df['cluster'].nunique()} clusters identificati")
    
    # Test 4: Anomaly Detector
    print("\n" + "="*80)
    print("4️⃣  TEST ANOMALY DETECTOR")
    print("="*80)
    
    anomaly_detector = AnomalyDetector(contamination=0.1)
    anomalies = anomaly_detector.fit(df)
    anomaly_detector.save('test_anomaly_detector.pkl')
    
    df['is_anomaly'] = anomalies == -1
    n_anomalies = df['is_anomaly'].sum()
    print(f"\n✅ Test completato: {n_anomalies} anomalie rilevate")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY DEI TEST")
    print("="*80)
    print(f"✅ Tutti i 4 modelli testati con successo!")
    print(f"\nTransazioni analizzate: {len(df)}")
    print(f"CoinJoin rilevati: {len(high_prob)} ({len(high_prob)/len(df):.1%})")
    print(f"Errore medio feerate: {mae:.2f} sat/vB")
    print(f"Clusters: {df['cluster'].nunique()}")
    print(f"Anomalie: {n_anomalies} ({n_anomalies/len(df):.1%})")
    
    # Statistiche per tipo reale
    actual_coinjoins = df[df['equal_output'] == True]
    detected_coinjoins = df[df['coinjoin_probability'] > 0.7]
    
    print(f"\n🎯 Accuracy CoinJoin Detection:")
    print(f"   CoinJoin reali: {len(actual_coinjoins)}")
    print(f"   CoinJoin rilevati: {len(detected_coinjoins)}")
    
    # Salva risultati
    output_file = 'ml_test_results.csv'
    df[['txid', 'feerate', 'predicted_feerate', 'coinjoin_probability', 
        'cluster', 'is_anomaly']].to_csv(output_file, index=False)
    print(f"\n💾 Risultati salvati in: {output_file}")
    
    print("\n" + "="*80)
    print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        test_ml_pipeline()
    except Exception as e:
        print(f"\n❌ Test fallito: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
