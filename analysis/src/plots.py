import matplotlib.pyplot as plt
import seaborn as sns
import os

def setup_plotting_style():
    """Configura lo stile dei grafici."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.dpi'] = 100

def plot_io_distribution(df_tx, output_dir):
    """Genera grafico distribuzione input/output."""
    print("\nGenerazione grafico distribuzione I/O...")
    
    # Top 10 pattern
    io_dist = df_tx.groupby(['inputs_count', 'outputs_count']).size().reset_index(name='count')
    io_dist['label'] = io_dist['inputs_count'].astype(str) + '→' + io_dist['outputs_count'].astype(str)
    io_dist = io_dist.sort_values('count', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(io_dist['label'], io_dist['count'], color='steelblue', edgecolor='black', alpha=0.7)
    
    # Percentuali sopra le barre
    for bar, count in zip(bars, io_dist['count']):
        height = bar.get_height()
        percentage = count / len(df_tx) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Pattern Input → Output', fontsize=13, fontweight='bold')
    ax.set_ylabel('Numero Transazioni', fontsize=13, fontweight='bold')
    ax.set_title('Distribuzione Pattern Input/Output (Top 10)', fontsize=15, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'fig1_io_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")

def plot_heuristics_pie(df_tx, output_dir):
    """Genera grafico a torta per euristiche."""
    print("\nGenerazione grafico euristiche...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RBF
    rbf_data = [df_tx['is_rbf'].sum(), len(df_tx) - df_tx['is_rbf'].sum()]
    colors1 = ['#ff6b6b', '#95e1d3']
    explode1 = (0.1, 0)
    
    ax1.pie(rbf_data, labels=['RBF Enabled', 'RBF Disabled'], 
            autopct='%1.1f%%', startangle=90, colors=colors1, explode=explode1,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Replace-By-Fee (RBF) Adoption', fontsize=14, fontweight='bold')
    
    # Equal Output
    equal_data = [df_tx['equal_output'].sum(), len(df_tx) - df_tx['equal_output'].sum()]
    colors2 = ['#feca57', '#48dbfb']
    explode2 = (0.1, 0)
    
    ax2.pie(equal_data, labels=['Equal Outputs', 'Mixed Outputs'],
            autopct='%1.1f%%', startangle=90, colors=colors2, explode=explode2,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Equal Output Pattern (CoinJoin Indicator)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig2_heuristics_pie.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")

def plot_coinjoin_distribution(df_tx, output_dir):
    """Genera istogramma distribuzione CoinJoin Score."""
    print("\nGenerazione grafico CoinJoin score...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Istogramma
    n, bins, patches = ax.hist(df_tx['coinjoin_score'], bins=50, 
                                color='purple', alpha=0.7, edgecolor='black')
    
    # Colora diversamente le barre con score > 0.7
    for i, patch in enumerate(patches):
        if bins[i] > 0.7:
            patch.set_facecolor('red')
            patch.set_alpha(0.9)
    
    # Linea mediana
    median = df_tx['coinjoin_score'].median()
    ax.axvline(median, color='green', linestyle='--', linewidth=2, 
               label=f'Mediana: {median:.3f}')
    
    # Linea soglia CoinJoin
    ax.axvline(0.7, color='red', linestyle='--', linewidth=2,
               label='Soglia CoinJoin: 0.7')
    
    ax.set_xlabel('CoinJoin Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Numero Transazioni', fontsize=13, fontweight='bold')
    ax.set_title('Distribuzione CoinJoin Score', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig3_coinjoin_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")

def plot_tx_size_distribution(df_tx, output_dir):
    """Genera istogramma distribuzione dimensione transazioni."""
    print("\nGenerazione grafico dimensione transazioni...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Size distribution
    ax1.hist(df_tx['size'], bins=100, color='teal', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Size (bytes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Numero Transazioni', fontsize=12, fontweight='bold')
    ax1.set_title('Distribuzione Dimensione Transazioni', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, df_tx['size'].quantile(0.95))  # Zoom su 95% dei dati
    
    # VSize distribution
    ax2.hist(df_tx['vsize'], bins=100, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('VSize (vbytes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Numero Transazioni', fontsize=12, fontweight='bold')
    ax2.set_title('Distribuzione Virtual Size (SegWit)', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, df_tx['vsize'].quantile(0.95))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig4_tx_size_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")

def plot_blocks_timeline(df_tx, output_dir):
    """Genera scatter plot timeline blocchi analizzati."""
    print("\nGenerazione timeline blocchi...")
    
    # Raggruppa per blocco
    blocks = df_tx.groupby('block_height').size().reset_index(name='tx_count')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.scatter(blocks['block_height'], blocks['tx_count'], 
               alpha=0.5, s=10, color='darkblue')
    
    ax.set_xlabel('Block Height', fontsize=13, fontweight='bold')
    ax.set_ylabel('Transazioni per Blocco', fontsize=13, fontweight='bold')
    ax.set_title('Distribuzione Temporale Blocchi Analizzati', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Annotazione
    ax.text(0.02, 0.98, f'Blocchi totali: {len(blocks):,}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig5_blocks_timeline.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")
