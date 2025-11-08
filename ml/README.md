# Machine Learning per Analisi Blockchain

Questo modulo implementa diversi modelli di Machine Learning per analizzare pattern nella blockchain Bitcoin.

## 🤖 Modelli Implementati

### 1. **CoinJoin Detector** 
- **Tipo**: Random Forest Classifier
- **Scopo**: Rilevare transazioni CoinJoin con alta precisione
- **Features**: vsize, inputs/outputs, feerate, coinjoin_score, ratios
- **Output**: Probabilità che una transazione sia un CoinJoin

### 2. **Feerate Predictor**
- **Tipo**: Gradient Boosting Regressor
- **Scopo**: Predire il feerate ottimale per conferma veloce
- **Features**: inputs/outputs count, vsize, ora del giorno, giorno settimana, RBF
- **Output**: Feerate raccomandato in sat/vB

### 3. **Transaction Clusterer**
- **Tipo**: K-Means / DBSCAN
- **Scopo**: Raggruppare transazioni con pattern simili
- **Features**: size, feerate, I/O ratios, coinjoin_score
- **Output**: Cluster assignment per ogni transazione

### 4. **Anomaly Detector**
- **Tipo**: Isolation Forest
- **Scopo**: Rilevare transazioni anomale o sospette
- **Features**: size, fees, I/O counts e ratios
- **Output**: Flag anomalia (normale/-1 anomalo)

## 📦 Installazione

```bash
cd ml
pip install -r requirements.txt
```

## 🚀 Utilizzo

### Training dei modelli

Prima esecuzione - addestra tutti i modelli:

```bash
python train_models.py
```

Questo:
1. Carica dati dal database PostgreSQL
2. Esegue feature engineering
3. Addestra 4 modelli diversi
4. Salva i modelli nella directory `models/`
5. Mostra metriche di performance

**Requisiti**: Almeno 1000 transazioni nel database per training efficace.

### Analisi transazioni

Analizza le transazioni più recenti con tutti i modelli:

```bash
python predict.py --mode analyze
```

Output:
- Lista CoinJoin candidates con probabilità
- Confronto feerate attuale vs predetto
- Distribuzione per cluster
- Anomalie rilevate
- File CSV con risultati in `results/ml_predictions.csv`

### Predizione feerate ottimale

Predici il feerate ideale per una nuova transazione:

```bash
# Esempio: 2 input, 2 output, 250 vB
python predict.py --mode predict_fee --inputs 2 --outputs 2 --vsize 250
```

Output:
```
Feerate ottimale predetto: 15.23 sat/vB
Fee totale stimata: 3808 sat
```

## 📊 Feature Engineering

Il modulo calcola automaticamente diverse features derivate:

- **Fee ratios**: fee_per_input, fee_per_output
- **Size ratios**: size_per_input, size_per_output
- **I/O ratio**: inputs_count / outputs_count
- **Temporal features**: hour, day_of_week, is_weekend
- **Weight ratio**: weight_to_size_ratio (SegWit indicator)

## 🔄 Ri-training periodico

I modelli dovrebbero essere ri-addestrati periodicamente per adattarsi a:
- Cambiamenti nelle fee del network
- Nuovi pattern di transazioni
- Evoluzione tecnologie (Taproot, Lightning)

Cron job consigliato:
```bash
# Ri-training settimanale
0 2 * * 0 cd /path/to/ml && python train_models.py >> train.log 2>&1
```

## 📈 Metriche di Performance

### CoinJoin Detector
- Precision: ~85-95% (dipende dai dati)
- Recall: ~75-90%
- F1-Score: ~80-92%

### Feerate Predictor
- R² Score: ~0.70-0.85
- RMSE: 2-5 sat/vB (network calmo)

### Transaction Clusterer
- Silhouette Score: 0.3-0.6
- 5 clusters principali identificati

### Anomaly Detector
- Contamination: 5% (configurabile)
- False Positive Rate: <10%

## 🛠️ Integrazione con Ingest

Per analisi in real-time, puoi integrare i modelli nel processo di ingest:

```python
from ml.train_models import CoinJoinDetector, AnomalyDetector

# Carica modelli pre-addestrati
coinjoin_detector = CoinJoinDetector.load()
anomaly_detector = AnomalyDetector.load()

# Durante ingest...
for tx in new_transactions:
    # Feature engineering
    features = engineer_features(tx)
    
    # Predizioni
    is_coinjoin = coinjoin_detector.predict(features)
    is_anomaly = anomaly_detector.predict(features)
    
    # Salva in database...
```

## 📁 Struttura Directory

```
ml/
├── train_models.py          # Training di tutti i modelli
├── predict.py               # Predizioni e analisi
├── requirements.txt         # Dipendenze Python
├── README.md               # Questa guida
├── models/                 # Modelli salvati (generati)
│   ├── coinjoin_detector.pkl
│   ├── feerate_predictor.pkl
│   ├── transaction_clusterer.pkl
│   └── anomaly_detector.pkl
└── results/                # Output analisi (generati)
    └── ml_predictions.csv
```

## 🔬 Esempi d'uso avanzati

### Analisi batch personalizzata

```python
from train_models import load_transaction_data, engineer_features
from train_models import CoinJoinDetector

# Carica e prepara dati
df = load_transaction_data(min_samples=5000)
df = engineer_features(df)

# Filtra per blocchi specifici
df_filtered = df[df['block_height'] > 800000]

# Analisi CoinJoin
detector = CoinJoinDetector.load()
probabilities = detector.predict(df_filtered)

# Top 10 CoinJoin
df_filtered['coinjoin_prob'] = probabilities
top_coinjoins = df_filtered.nlargest(10, 'coinjoin_prob')
print(top_coinjoins[['txid', 'coinjoin_prob', 'inputs_count', 'outputs_count']])
```

### Grid Search per ottimizzazione

```python
from sklearn.model_selection import GridSearchCV

# Parametri da testare
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Training
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

## ⚠️ Limitazioni

1. **Dati richiesti**: Servono almeno 1000 transazioni per training efficace
2. **Drift temporale**: Le fee cambiano nel tempo, serve ri-training
3. **Cold start**: Non può predire pattern mai visti prima
4. **Label quality**: CoinJoin detection dipende dall'euristica iniziale
5. **Resource intensive**: Training richiede CPU e memoria

## 🎯 Prossimi sviluppi

- [ ] Modello LSTM per predizione time-series del mempool
- [ ] Deep Learning per pattern più complessi
- [ ] Address clustering con Graph Neural Networks
- [ ] Reinforcement Learning per fee optimization
- [ ] Transfer learning da altri blockchain
- [ ] Model explainability con SHAP values

## 📚 Riferimenti

- Scikit-learn Documentation: https://scikit-learn.org/
- Bitcoin ML Research: https://arxiv.org/abs/2003.14389
- Chain Analysis Techniques: https://cseweb.ucsd.edu/~smeiklejohn/files/imc13.pdf
