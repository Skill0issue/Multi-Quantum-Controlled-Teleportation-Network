"""
plotting.py — Publication-quality figures.

Figure 1: Fidelity before/after recovery vs number of Byzantine nodes
Figure 2: Distribution of fidelity improvement
Figure 3: Detection rate by strategy type
Figure 4: M correction accuracy vs noise level
Figure 5: Fidelity vs noise level
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger("plotting")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
COLORS = sns.color_palette("muted")
DPI, FS = 150, (7, 5)

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path}")


def plot_fidelity_vs_byzantine(df_byz: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=FS)
    groups = sorted(df_byz["sweep_f"].unique())
    x = np.arange(len(groups)); w = 0.35
    mb = [df_byz[df_byz.sweep_f==f].fidelity_before.mean() for f in groups]
    ma = [df_byz[df_byz.sweep_f==f].fidelity_after.mean()  for f in groups]
    sb = [df_byz[df_byz.sweep_f==f].fidelity_before.std()  for f in groups]
    sa = [df_byz[df_byz.sweep_f==f].fidelity_after.std()   for f in groups]
    ax.bar(x-w/2, mb, w, yerr=sb, label="Before recovery", color=COLORS[0], capsize=4, alpha=0.85)
    ax.bar(x+w/2, ma, w, yerr=sa, label="After recovery",  color=COLORS[1], capsize=4, alpha=0.85)
    ax.set_xlabel("Number of Byzantine Nodes (f)")
    ax.set_ylabel("Teleportation Fidelity")
    ax.set_title("Fidelity vs Byzantine Nodes")
    ax.set_xticks(x); ax.set_xticklabels([str(f) for f in groups])
    ax.set_ylim(0, 1.05); ax.axhline(1, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.legend()
    _save(fig, out_dir / "fig1_fidelity_vs_byzantine.png")


def plot_fidelity_improvement(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=FS)
    imp = df.fidelity_after - df.fidelity_before
    ax.hist(imp, bins=40, color=COLORS[2], edgecolor="white", alpha=0.85)
    ax.axvline(imp.mean(), color="crimson", ls="--", lw=1.5,
               label=f"Mean = {imp.mean():.4f}")
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Fidelity Improvement (After − Before)")
    ax.set_ylabel("Trial Count")
    ax.set_title("Fidelity Improvement from Byzantine Recovery")
    ax.legend()
    _save(fig, out_dir / "fig2_fidelity_improvement.png")


def plot_detection_by_strategy(df: pd.DataFrame, out_dir: Path):
    """Bar chart: detection rate broken down by strategy type."""
    fig, ax = plt.subplots(figsize=FS)
    # Per-trial detection rate
    def det_rate(row):
        byz = set(eval(row["byzantine_nodes"]))
        det = set(eval(row["detected_nodes"]))
        if not byz: return 1.0
        return len(byz & det) / len(byz)

    df = df.copy()
    df["det_rate"] = df.apply(det_rate, axis=1)
    # Group by dominant strategy
    df["dom_strategy"] = df.apply(
        lambda r: "coherent_flip" if r["n_coherent_flip"] > 0
        else ("equivocate" if r["n_equivocate"] > 0 else "withhold"),
        axis=1
    )
    rates = df.groupby("dom_strategy")["det_rate"].mean()
    ax.bar(rates.index, rates.values, color=COLORS[3], alpha=0.85)
    ax.set_ylabel("Mean Detection Rate")
    ax.set_title("Detection Rate by Dominant Byzantine Strategy")
    ax.set_ylim(0, 1.05)
    ax.axhline(1, color="gray", ls="--", lw=0.8, alpha=0.6)
    _save(fig, out_dir / "fig3_detection_by_strategy.png")


def plot_M_accuracy_vs_noise(df_noise: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=FS)
    groups = sorted(df_noise["sweep_noise"].unique())
    labels = [f"{n*100:.1f}%" for n in groups]
    mb = [df_noise[df_noise.sweep_noise==ns].M_correct_before.mean() for ns in groups]
    ma = [df_noise[df_noise.sweep_noise==ns].M_correct_after.mean()  for ns in groups]
    ax.plot(labels, mb, "o--", color=COLORS[0], lw=1.5, ms=7, label="Before recovery")
    ax.plot(labels, ma, "s-",  color=COLORS[1], lw=1.5, ms=7, label="After recovery")
    ax.set_xlabel("Noise Level"); ax.set_ylabel("M Correction Accuracy")
    ax.set_title("M Accuracy vs Noise Level")
    ax.set_ylim(0, 1.05); ax.legend()
    _save(fig, out_dir / "fig4_M_accuracy_vs_noise.png")


def plot_fidelity_vs_noise(df_noise: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=FS)
    groups = sorted(df_noise["sweep_noise"].unique())
    labels = [f"{n*100:.1f}%" for n in groups]
    mb = [df_noise[df_noise.sweep_noise==ns].fidelity_before.mean() for ns in groups]
    ma = [df_noise[df_noise.sweep_noise==ns].fidelity_after.mean()  for ns in groups]
    sb = [df_noise[df_noise.sweep_noise==ns].fidelity_before.std()  for ns in groups]
    sa = [df_noise[df_noise.sweep_noise==ns].fidelity_after.std()   for ns in groups]
    ax.errorbar(labels, mb, yerr=sb, fmt="o--", color=COLORS[0], capsize=4,
                lw=1.5, ms=7, label="Before recovery")
    ax.errorbar(labels, ma, yerr=sa, fmt="s-",  color=COLORS[1], capsize=4,
                lw=1.5, ms=7, label="After recovery")
    ax.set_xlabel("Noise Level"); ax.set_ylabel("Mean Fidelity")
    ax.set_title("Fidelity vs Noise Level")
    ax.set_ylim(0, 1.05); ax.axhline(1, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.legend()
    _save(fig, out_dir / "fig5_fidelity_vs_noise.png")


def generate_all_plots(df_default, df_byz, df_noise, out_dir: Path):
    log.info("Generating plots…")
    plot_fidelity_vs_byzantine(df_byz, out_dir)
    plot_fidelity_improvement(df_default, out_dir)
    plot_detection_by_strategy(df_default, out_dir)
    plot_M_accuracy_vs_noise(df_noise, out_dir)
    plot_fidelity_vs_noise(df_noise, out_dir)
    log.info("All plots saved.")
