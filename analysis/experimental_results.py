"""
Experimental Results Generation for Bitcoin Chain Analysis Thesis
Author: Luca Impellizzeri
University of Catania - Computer Engineering

This script generates statistical analysis and visualizations for 4 case studies:
1. Replace-By-Fee (RBF) Analysis
2. CoinJoin Detection
3. Fee Rate Distribution & Anomalies
4. Change Address Detection Patterns

Requirements:
- PostgreSQL database with collected blockchain data
- Python packages: pandas, matplotlib, seaborn, sqlalchemy, numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import json
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = 'analysis/results'
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'blockchain',
    'user': 'postgres',
    'password': 'postgres'
}

# Plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def connect_database():
    """Establish database connection"""
    try:
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

def check_data_availability(engine):
    """Check if we have enough data for analysis"""
    query = "SELECT COUNT(*) as count FROM tx_basic"
    result = pd.read_sql(query, engine)
    count = result['count'][0]
    
    print(f"\n📊 Dataset Status:")
    print(f"   Total transactions: {count:,}")
    
    if count < 100:
        print(f"\n⚠️  WARNING: Only {count} transactions available.")
        print(f"   Recommendation: Let the system collect more data (target: 10,000+)")
        print(f"   Continue anyway? (y/n): ", end='')
        
        choice = input().lower()
        if choice != 'y':
            print("Exiting. Run this script again when more data is available.")
            sys.exit(0)
    
    return count

def analyze_rbf(engine):
    """Case Study 1: Replace-By-Fee Analysis"""
    print("\n" + "="*80)
    print("CASE STUDY 1: Replace-By-Fee (RBF) Analysis")
    print("="*80)
    
    query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN is_rbf THEN 1 ELSE 0 END) as rbf_count
        FROM tx_heuristics
    """
    
    stats = pd.read_sql(query, engine).iloc[0]
    rbf_percentage = (stats['rbf_count'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    print(f"\n Results:")
    print(f"   Total transactions: {stats['total']:,}")
    print(f"   RBF transactions: {stats['rbf_count']:,} ({rbf_percentage:.2f}%)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    labels = ['Non-RBF', 'RBF']
    sizes = [stats['total'] - stats['rbf_count'], stats['rbf_count']]
    colors = ['#3498db', '#e74c3c']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('RBF Distribution', fontweight='bold')
    
    # Comparison bar chart
    query_comparison = """
        SELECT 
            h.is_rbf,
            AVG(b.num_inputs) as avg_inputs,
            AVG(b.num_outputs) as avg_outputs,
            AVG(b.fee_rate) as avg_fee_rate
        FROM tx_heuristics h
        JOIN tx_basic b ON h.txid = b.txid
        WHERE b.fee_rate > 0
        GROUP BY h.is_rbf
    """
    
    comparison = pd.read_sql(query_comparison, engine)
    if len(comparison) > 0:
        comparison['type'] = comparison['is_rbf'].map({True: 'RBF', False: 'Non-RBF'})
        
        x = np.arange(len(comparison))
        width = 0.25
        
        ax2.bar(x - width, comparison['avg_inputs'], width, label='Avg Inputs', color='#3498db')
        ax2.bar(x, comparison['avg_outputs'], width, label='Avg Outputs', color='#2ecc71')
        ax2.bar(x + width, comparison['avg_fee_rate'], width, label='Avg Fee Rate', color='#e74c3c')
        
        ax2.set_xlabel('Transaction Type')
        ax2.set_ylabel('Value')
        ax2.set_title('RBF vs Non-RBF Characteristics', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison['type'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/case1_rbf_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR}/case1_rbf_analysis.png")
    plt.close()
    
    return {
        'total': int(stats['total']),
        'rbf_count': int(stats['rbf_count']),
        'rbf_percentage': float(rbf_percentage)
    }

def analyze_coinjoin(engine):
    """Case Study 2: CoinJoin Detection"""
    print("\n" + "="*80)
    print("CASE STUDY 2: CoinJoin Detection")
    print("="*80)
    
    query = """
        SELECT coinjoin_score, has_equal_outputs, equal_outputs_count
        FROM tx_heuristics
        WHERE coinjoin_score > 0
    """
    
    data = pd.read_sql(query, engine)
    
    if len(data) == 0:
        print("\n⚠️  No transactions with CoinJoin score > 0 found")
        return None
    
    # Statistics
    high_score = (data['coinjoin_score'] > 0.7).sum()
    medium_score = ((data['coinjoin_score'] > 0.5) & (data['coinjoin_score'] <= 0.7)).sum()
    low_score = ((data['coinjoin_score'] > 0.3) & (data['coinjoin_score'] <= 0.5)).sum()
    
    print(f"\n Results:")
    print(f"   High probability (>0.7): {high_score:,}")
    print(f"   Medium probability (0.5-0.7): {medium_score:,}")
    print(f"   Low probability (0.3-0.5): {low_score:,}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax1.hist(data['coinjoin_score'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High threshold')
    ax1.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Medium threshold')
    ax1.set_xlabel('CoinJoin Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('CoinJoin Score Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Equal outputs analysis
    equal_counts = data['equal_outputs_count'].value_counts().sort_index()
    ax2.bar(equal_counts.index, equal_counts.values, color='#16a085', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Equal Outputs')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Equal Outputs Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/case2_coinjoin_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR}/case2_coinjoin_analysis.png")
    plt.close()
    
    return {
        'high_score_count': int(high_score),
        'medium_score_count': int(medium_score),
        'low_score_count': int(low_score)
    }

def analyze_fees(engine):
    """Case Study 3: Fee Rate Analysis"""
    print("\n" + "="*80)
    print("CASE STUDY 3: Fee Rate Analysis")
    print("="*80)
    
    query = """
        SELECT fee_rate, num_inputs, num_outputs
        FROM tx_basic
        WHERE fee_rate > 0 AND fee_rate < 1000
    """
    
    data = pd.read_sql(query, engine)
    
    if len(data) == 0:
        print("\n⚠️  No valid fee data available")
        return None
    
    # Statistics
    print(f"\n Results:")
    print(f"   Mean fee rate: {data['fee_rate'].mean():.2f} sat/vB")
    print(f"   Median fee rate: {data['fee_rate'].median():.2f} sat/vB")
    print(f"   Std deviation: {data['fee_rate'].std():.2f} sat/vB")
    print(f"   Min: {data['fee_rate'].min():.2f} sat/vB")
    print(f"   Max: {data['fee_rate'].max():.2f} sat/vB")
    
    # Anomaly detection (IQR method)
    Q1 = data['fee_rate'].quantile(0.25)
    Q3 = data['fee_rate'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = ((data['fee_rate'] < lower_bound) | (data['fee_rate'] > upper_bound)).sum()
    anomaly_pct = (anomalies / len(data)) * 100
    
    print(f"\n Anomalies detected: {anomalies:,} ({anomaly_pct:.2f}%)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    ax1.boxplot(data['fee_rate'])
    ax1.set_ylabel('Fee Rate (sat/vB)')
    ax1.set_title('Fee Rate Distribution (Boxplot)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(data['fee_rate'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.axvline(x=data['fee_rate'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(x=data['fee_rate'].median(), color='blue', linestyle='--', linewidth=2, label='Median')
    ax2.set_xlabel('Fee Rate (sat/vB)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Fee Rate Distribution (Histogram)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/case3_fee_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR}/case3_fee_analysis.png")
    plt.close()
    
    return {
        'mean': float(data['fee_rate'].mean()),
        'median': float(data['fee_rate'].median()),
        'std': float(data['fee_rate'].std()),
        'anomalies': int(anomalies),
        'anomaly_percentage': float(anomaly_pct)
    }

def analyze_change_detection(engine):
    """Case Study 4: Change Address Detection"""
    print("\n" + "="*80)
    print("CASE STUDY 4: Change Address Detection")
    print("="*80)
    
    query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN change_output_index IS NOT NULL THEN 1 ELSE 0 END) as with_change
        FROM tx_heuristics
    """
    
    stats = pd.read_sql(query, engine).iloc[0]
    change_pct = (stats['with_change'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    print(f"\n Results:")
    print(f"   Total transactions: {stats['total']:,}")
    print(f"   With identified change: {stats['with_change']:,} ({change_pct:.2f}%)")
    
    # Output distribution
    query_outputs = """
        SELECT num_outputs, COUNT(*) as count
        FROM tx_basic
        GROUP BY num_outputs
        ORDER BY num_outputs
        LIMIT 15
    """
    
    output_dist = pd.read_sql(query_outputs, engine)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Change detection pie
    labels = ['No Change Identified', 'Change Identified']
    sizes = [stats['total'] - stats['with_change'], stats['with_change']]
    colors = ['#95a5a6', '#e67e22']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Change Detection Success Rate', fontweight='bold')
    
    # Output distribution
    ax2.bar(output_dist['num_outputs'], output_dist['count'], color='#34495e', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Outputs')
    ax2.set_ylabel('Number of Transactions')
    ax2.set_title('Transaction Output Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/case4_change_detection.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR}/case4_change_detection.png")
    plt.close()
    
    return {
        'total': int(stats['total']),
        'with_change': int(stats['with_change']),
        'percentage': float(change_pct)
    }

def generate_summary_report(engine, results):
    """Generate JSON summary report"""
    
    # Dataset info
    query_dataset = """
        SELECT 
            MIN(block_height) as min_block,
            MAX(block_height) as max_block,
            COUNT(DISTINCT block_height) as unique_blocks
        FROM tx_basic
    """
    
    dataset_info = pd.read_sql(query_dataset, engine).iloc[0]
    
    summary = {
        'generated_at': datetime.now().isoformat(),
        'dataset': {
            'total_transactions': results.get('total_transactions', 0),
            'min_block': int(dataset_info['min_block']) if dataset_info['min_block'] else 0,
            'max_block': int(dataset_info['max_block']) if dataset_info['max_block'] else 0,
            'unique_blocks': int(dataset_info['unique_blocks']) if dataset_info['unique_blocks'] else 0
        },
        'case_studies': {
            'rbf_analysis': results.get('rbf', {}),
            'coinjoin_detection': results.get('coinjoin', {}),
            'fee_analysis': results.get('fees', {}),
            'change_detection': results.get('change', {})
        }
    }
    
    with open(f'{OUTPUT_DIR}/experimental_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Saved: {OUTPUT_DIR}/experimental_results.json")
    return summary

def main():
    """Main execution"""
    print("="*80)
    print("EXPERIMENTAL RESULTS GENERATION")
    print("Bitcoin Chain Analysis - Master Thesis")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Connect to database
    engine = connect_database()
    print("✅ Database connected")
    
    # Check data availability
    total_tx = check_data_availability(engine)
    
    # Run analyses
    results = {'total_transactions': total_tx}
    
    try:
        results['rbf'] = analyze_rbf(engine)
        results['coinjoin'] = analyze_coinjoin(engine)
        results['fees'] = analyze_fees(engine)
        results['change'] = analyze_change_detection(engine)
        
        # Generate summary
        generate_summary_report(engine, results)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nGenerated files in '{OUTPUT_DIR}/':")
        print("  - case1_rbf_analysis.png")
        print("  - case2_coinjoin_analysis.png")
        print("  - case3_fee_analysis.png")
        print("  - case4_change_detection.png")
        print("  - experimental_results.json")
        print("\nYou can now include these results in your thesis!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
