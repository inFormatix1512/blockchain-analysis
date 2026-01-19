#!/usr/bin/env python3
"""
Confronto tra ere storiche di Bitcoin.

Genera un riepilogo statistico per macro-ere e salva un CSV
con le metriche principali (volume, dimensioni, pattern CoinJoin/RBF).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analysis.src.data_loader import load_all_data


ERAS = [
    ("2011-2015", 2011, 2015),
    ("2016-2019", 2016, 2019),
    ("2020-2023", 2020, 2023),
]


def _summarize_era(df: pd.DataFrame, label: str, start_year: int, end_year: int) -> dict:
    subset = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    if subset.empty:
        return {
            "era": label,
            "tx_count": 0,
            "avg_size": 0.0,
            "avg_vsize": 0.0,
            "avg_inputs": 0.0,
            "avg_outputs": 0.0,
            "coinjoin_pct": 0.0,
            "rbf_pct": 0.0,
        }

    return {
        "era": label,
        "tx_count": int(len(subset)),
        "avg_size": float(subset["size"].mean()),
        "avg_vsize": float(subset["vsize"].mean()),
        "avg_inputs": float(subset["inputs_count"].mean()),
        "avg_outputs": float(subset["outputs_count"].mean()),
        "coinjoin_pct": float(subset["equal_output"].mean() * 100),
        "rbf_pct": float(subset["is_rbf"].mean() * 100),
    }


def _plot_bar(df_summary: pd.DataFrame, column: str, title: str, ylabel: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_summary["era"], df_summary[column], color="#3b82f6")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Era")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"analysis/results/compare_eras_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    df_tx, _ = load_all_data(limit=None)
    if df_tx.empty:
        print("[ERRORE] Nessun dato disponibile nel database.")
        return

    df_tx = df_tx.copy()
    df_tx["year"] = pd.to_datetime(df_tx["ts"], unit="s").dt.year

    summary_rows = [_summarize_era(df_tx, label, start, end) for label, start, end in ERAS]
    df_summary = pd.DataFrame(summary_rows)

    csv_path = os.path.join(output_dir, "compare_eras_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"[OK] Salvato riepilogo: {csv_path}")

    _plot_bar(
        df_summary,
        "tx_count",
        "Numero di transazioni per era",
        "Transazioni",
        os.path.join(output_dir, "era_tx_count.png"),
    )
    _plot_bar(
        df_summary,
        "coinjoin_pct",
        "Percentuale CoinJoin (equal output) per era",
        "% CoinJoin",
        os.path.join(output_dir, "era_coinjoin_pct.png"),
    )
    _plot_bar(
        df_summary,
        "rbf_pct",
        "Percentuale RBF per era",
        "% RBF",
        os.path.join(output_dir, "era_rbf_pct.png"),
    )

    print("[OK] Grafici salvati in:", output_dir)


if __name__ == "__main__":
    main()
