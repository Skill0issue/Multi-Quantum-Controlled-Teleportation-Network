"""
plotting.py — Publication-quality figures for EVBTCST.

Figures from inherited byzantine_teleportation:
  Fig 1: Fidelity before/after VSS recovery vs Byzantine count
  Fig 2: Fidelity improvement distribution
  Fig 3: Detection rate by strategy
  Fig 4: M correction accuracy vs noise
  Fig 5: Fidelity vs noise

New EVBTCST figures:
  Fig 6: Detection power vs k (Table 2) — empirical vs theoretical
  Fig 7: M failure rate before/after VSS/Bell verification (Table 3)
  Fig 8: F_noise and likelihood ratio vs noise level (Table 4)
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


# ── Inherited figures (unchanged from byzantine_teleportation) ────────────────

def plot_fidelity_vs_byzantine(df_byz: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=FS)
    groups = sorted(df_byz["sweep_f"].unique())
    x = np.arange(len(groups)); w = 0.35
    mb = [df_byz[df_byz.sweep_f==f].fidelity_before.mean() for f in groups]
    ma = [df_byz[df_byz.sweep_f==f].fidelity_after.mean()  for f in groups]
    sb = [df_byz[df_byz.sweep_f==f].fidelity_before.std()  for f in groups]
    sa = [df_byz[df_byz.sweep_f==f].fidelity_after.std()   for f in groups]
    ax.bar(x-w/2, mb, w, yerr=sb, label="Before VSS recovery", color=COLORS[0], capsize=4, alpha=0.85)
    ax.bar(x+w/2, ma, w, yerr=sa, label="After VSS recovery",  color=COLORS[1], capsize=4, alpha=0.85)
    ax.set_xlabel("Number of Byzantine Nodes (f)")
    ax.set_ylabel("Teleportation Fidelity")
    ax.set_title("Fig 1: Fidelity vs Byzantine Nodes (VSS only)")
    ax.set_xticks(x); ax.set_xticklabels([str(f) for f in groups])
    ax.set_ylim(0, 1.05); ax.axhline(1, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.legend()
    _save(fig, out_dir / "fig1_fidelity_vs_byzantine.png")


def plot_fidelity_improvement(df: pd.DataFrame, out_dir: Path):
    reg = df[df["trial_type"] == "regular"] if "trial_type" in df.columns else df
    fig, ax = plt.subplots(figsize=FS)
    imp = reg.fidelity_after - reg.fidelity_before
    ax.hist(imp, bins=40, color=COLORS[2], edgecolor="white", alpha=0.85)
    ax.axvline(imp.mean(), color="crimson", ls="--", lw=1.5,
               label=f"Mean = {imp.mean():.4f}")
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Fidelity Improvement (After − Before)")
    ax.set_ylabel("Trial Count")
    ax.set_title("Fig 2: Fidelity Improvement from VSS Recovery")
    ax.legend()
    _save(fig, out_dir / "fig2_fidelity_improvement.png")


def plot_detection_by_strategy(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=FS)
    def det_rate(row):
        byz = set(eval(row["byzantine_nodes"]))
        det = set(eval(row["detected_nodes"]))
        if not byz: return 1.0
        return len(byz & det) / len(byz)
    df = df.copy()
    df["det_rate"] = df.apply(det_rate, axis=1)
    df["dom_strategy"] = df.apply(
        lambda r: "coherent_flip" if r["n_coherent_flip"] > 0
        else ("equivocate" if r["n_equivocate"] > 0 else "withhold"), axis=1
    )
    rates = df.groupby("dom_strategy")["det_rate"].mean()
    ax.bar(rates.index, rates.values, color=COLORS[3], alpha=0.85)
    ax.set_ylabel("Mean Detection Rate (VSS)")
    ax.set_title("Fig 3: VSS Detection Rate by Byzantine Strategy")
    ax.set_ylim(0, 1.05)
    ax.axhline(1, color="gray", ls="--", lw=0.8, alpha=0.6)
    _save(fig, out_dir / "fig3_detection_by_strategy.png")


def plot_M_accuracy_vs_noise(df_noise: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=FS)
    groups = sorted(df_noise["sweep_noise"].unique())
    labels = [f"{n*100:.1f}%" for n in groups]
    mb = [df_noise[df_noise.sweep_noise==ns].M_correct_before.mean() for ns in groups]
    ma = [df_noise[df_noise.sweep_noise==ns].M_correct_after.mean()  for ns in groups]
    ax.plot(labels, mb, "o--", color=COLORS[0], lw=1.5, ms=7, label="Before VSS recovery")
    ax.plot(labels, ma, "s-",  color=COLORS[1], lw=1.5, ms=7, label="After VSS recovery")
    ax.set_xlabel("Noise Level"); ax.set_ylabel("M Correction Accuracy")
    ax.set_title("Fig 4: M Accuracy vs Noise (VSS only)")
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
                lw=1.5, ms=7, label="Before VSS recovery")
    ax.errorbar(labels, ma, yerr=sa, fmt="s-",  color=COLORS[1], capsize=4,
                lw=1.5, ms=7, label="After VSS recovery")
    ax.set_xlabel("Noise Level"); ax.set_ylabel("Mean Fidelity")
    ax.set_title("Fig 5: Fidelity vs Noise (VSS only)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.legend()
    _save(fig, out_dir / "fig5_fidelity_vs_noise.png")


# ── New EVBTCST figures ───────────────────────────────────────────────────────

def plot_detection_power_vs_k(df_t2: pd.DataFrame, out_dir: Path):
    """Fig 6: Empirical vs theoretical detection power as function of k."""
    fig, ax = plt.subplots(figsize=FS)

    ax.plot(df_t2["k"], df_t2["empirical_power"],
            "o-", color=COLORS[0], lw=2, ms=7, label="Empirical (simulation)")
    ax.plot(df_t2["k"], df_t2["theoretical_power"],
            "s--", color=COLORS[1], lw=2, ms=7, label="Theoretical (Theorem 1)")

    ax.axhline(0.9999, color="crimson", ls=":", lw=1.5, label="Target 0.9999")
    ax.axvline(7,      color="gray",    ls=":", lw=1.5, label="k=7 (paper)")

    ax.set_xlabel("Verification Trials k")
    ax.set_ylabel("Detection Power (1 − β)")
    ax.set_title("Fig 6: Detection Power vs Verification Trials k\n"
                 "(Coherent-flip adversary, n=8, p=0.01, α=0.01)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, df_t2["k"].max() + 1)
    ax.legend(loc="lower right")
    _save(fig, out_dir / "fig6_detection_power_vs_k.png")


def plot_M_failure_before_after_bellverif(df_t3: pd.DataFrame, out_dir: Path):
    """Fig 7: M failure rate at three stages for f=0,1,2."""
    fig, ax = plt.subplots(figsize=FS)

    f_vals = df_t3["f"].values
    x = np.arange(len(f_vals))
    w = 0.25

    mb  = df_t3["M_fail_before"].values
    ma  = df_t3["M_fail_after_VSS"].values

    ax.bar(x - w, mb,  w, label="Before VSS recovery",   color=COLORS[0], alpha=0.85)
    ax.bar(x,     ma,  w, label="After VSS recovery",    color=COLORS[1], alpha=0.85)

    # Annotate XOR cancellation for f=2
    if len(df_t3[df_t3["f"]==2]) > 0:
        xor_rate = df_t3[df_t3["f"]==2]["xor_cancel_rate"].values[0]
        if xor_rate is not None:
            ax.annotate(
                f"XOR cancel\n≈{xor_rate:.1%}",
                xy=(x[-1] + w/2, ma[-1] + 0.02),
                fontsize=9, color="darkred", ha="center",
            )

    ax.set_xlabel("Byzantine Nodes (f)")
    ax.set_ylabel("M Correction Failure Rate")
    ax.set_title("Fig 7: M Failure Rate Before/After VSS Recovery\n"
                 "(n=8, p=0.01; coherent-flip undetectable by VSS)")
    ax.set_xticks(x); ax.set_xticklabels([str(f) for f in f_vals])
    ax.set_ylim(0, max(mb.max(), ma.max()) * 1.3 + 0.05)
    ax.axhline(0.508, color="crimson", ls=":", lw=1.2, alpha=0.7, label="Paper: 50.8% (f=2)")
    ax.legend()
    _save(fig, out_dir / "fig7_M_failure_before_after_bellverif.png")


def plot_noise_scaling_viability(df_t4: pd.DataFrame, out_dir: Path):
    """Fig 8: F_noise and likelihood ratio vs noise level, with viability threshold."""
    fig, ax1 = plt.subplots(figsize=FS)
    ax2 = ax1.twinx()

    labels = [f"{p*100:.1f}%" for p in df_t4["noise_p"]]
    x = np.arange(len(labels))

    ax1.plot(x, df_t4["F_noise_sim"],  "o-",  color=COLORS[0], lw=2, ms=7, label="F_noise (sim)")
    ax1.plot(x, df_t4["F_noise_theory"], "s--", color=COLORS[1], lw=1.5, ms=6, label="F_noise (theory)")
    ax1.axhline(0.75, color=COLORS[0], ls=":", lw=1, alpha=0.6, label="F=3/4 threshold")
    ax1.set_ylabel("Noise Fidelity F_noise", color=COLORS[0])
    ax1.set_ylim(0, 1.05)

    ax2.plot(x, df_t4["likelihood_ratio"], "^-", color=COLORS[3], lw=2, ms=7, label="Likelihood ratio")
    ax2.axhline(12, color="crimson", ls="--", lw=1.5, label="Viability threshold (12×)")
    ax2.set_ylabel("Detection Likelihood Ratio", color=COLORS[3])
    ax2.set_ylim(0, df_t4["likelihood_ratio"].max() * 1.2 + 2)

    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("Noise Level p")
    ax1.set_title("Fig 8: F_noise and Detection Likelihood Ratio vs Noise\n"
                  "(n=8, f=0; viability threshold at 12×)")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=9)
    _save(fig, out_dir / "fig8_noise_scaling_viability.png")


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_plots(
    df_default: pd.DataFrame,
    df_byz: pd.DataFrame,
    df_noise: pd.DataFrame,
    df_t2: pd.DataFrame,
    df_t3: pd.DataFrame,
    df_t4: pd.DataFrame,
    out_dir: Path,
):
    log.info("Generating all plots…")
    out_dir = Path(out_dir)

    # Inherited
    if df_byz is not None and "sweep_f" in df_byz.columns:
        plot_fidelity_vs_byzantine(df_byz, out_dir)
    if df_default is not None and len(df_default):
        plot_fidelity_improvement(df_default, out_dir)
        plot_detection_by_strategy(df_default, out_dir)
    if df_noise is not None and "sweep_noise" in df_noise.columns:
        plot_M_accuracy_vs_noise(df_noise, out_dir)
        plot_fidelity_vs_noise(df_noise, out_dir)

    # New EVBTCST
    if df_t2 is not None and len(df_t2):
        plot_detection_power_vs_k(df_t2, out_dir)
    if df_t3 is not None and len(df_t3):
        plot_M_failure_before_after_bellverif(df_t3, out_dir)
    if df_t4 is not None and len(df_t4):
        plot_noise_scaling_viability(df_t4, out_dir)

    log.info("All plots saved.")
