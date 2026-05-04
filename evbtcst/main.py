"""
main.py — EVBTCST protocol: full experiment orchestrator.

Usage
-----
  python main.py               # full run (500 trials, all tables)
  python main.py --quick       # fast smoke test (30 trials)
  python main.py --tables-only # skip plots
  python main.py --plots-only  # regenerate plots from saved CSVs
  python main.py --lemma1      # only run Lemma 1 check
  python main.py --table2      # only run Table 2
  python main.py --table3      # only run Table 3
  python main.py --table4      # only run Table 4

Output files  (results/)
------------
  default_results.csv
  sweep_byzantine.csv
  sweep_noise.csv
  table2_detection_power.csv
  table3_fidelity_byzantine.csv
  table4_noise_scaling.csv
  lemma1_table.csv
  fig1..fig8 *.png
"""

from __future__ import annotations
import argparse
import logging
import sys
import os
from pathlib import Path

import pandas as pd

# Ensure local imports work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SimConfig
from simulator import EVBTCSTSimulator
from analysis import compute_statistics, print_summary, xor_cancellation_rate
from plotting import generate_all_plots
from lemma1_check import run_lemma1_check
from experiments.table2_detection_power import run_table2
from experiments.table3_fidelity_vs_byzantine import run_table3
from experiments.table4_noise_scaling import run_table4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("simulation.log")],
)
# Suppress Qiskit's very verbose transpiler pass-level logging
for _noisy in ("qiskit.passmanager", "qiskit.compiler", "qiskit.transpiler"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("main")

OUT = Path("results")
OUT.mkdir(exist_ok=True)


# ── Individual experiment runners ────────────────────────────────────────────

def run_default(n_trials: int = 500) -> tuple[pd.DataFrame, dict]:
    log.info(f"=== Default run: n=8, f=2, p=0.01, {n_trials} trials ===")
    cfg = SimConfig(n_controllers=8, max_byzantine=2,
                    noise_level=0.01, n_trials=n_trials, delta=0.10)
    sim = EVBTCSTSimulator(cfg)
    df, sprt = sim.run()
    df.to_csv(OUT / "default_results.csv", index=False)
    return df, sprt


def run_sweep_byzantine(n_trials: int = 100) -> pd.DataFrame:
    log.info("=== Sweep Byzantine f=0,1,2 ===")
    frames = []
    for f in [0, 1, 2]:
        cfg = SimConfig(n_controllers=8, max_byzantine=f,
                        noise_level=0.01, n_trials=n_trials, delta=0.10)
        sim = EVBTCSTSimulator(cfg)
        df, _ = sim.run()
        df["sweep_f"] = f
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "sweep_byzantine.csv", index=False)
    return result


def run_sweep_noise(n_trials: int = 100) -> pd.DataFrame:
    log.info("=== Sweep Noise p=0,0.005,0.01,0.02 ===")
    frames = []
    for noise in [0.0, 0.005, 0.01, 0.02]:
        cfg = SimConfig(n_controllers=8, max_byzantine=2,
                        noise_level=noise, n_trials=n_trials, delta=0.10)
        sim = EVBTCSTSimulator(cfg)
        df, _ = sim.run()
        df["sweep_noise"] = noise
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "sweep_noise.csv", index=False)
    return result


def run_lemma1(n_controllers: int = 8) -> list[dict]:
    log.info("=== Lemma 1 Check ===")
    results = run_lemma1_check(n_controllers=n_controllers, verbose=True)
    df = pd.DataFrame(results)
    df.to_csv(OUT / "lemma1_table.csv", index=False)
    return results


def print_full_summary(df: pd.DataFrame, sprt: dict, label: str = "default"):
    reg = df[df["trial_type"] == "regular"]
    ver = df[df["trial_type"] == "verification"]

    stats = compute_statistics(df)
    print_summary(stats)

    print(f"\n  SPRT Summary ({label})")
    print(f"  {'controller':<14} {'LLR':>8} {'flips':>7} {'seen':>6} {'flagged':>8}")
    print(f"  {'-'*46}")
    for k, s in sorted(sprt.items()):
        print(f"  C{k:<13} {s['llr']:>8.3f} {s['flips']:>7} {s['seen']:>6} {str(s['flagged']):>8}")

    flagged = {k for k, s in sprt.items() if s["flagged"]}
    print(f"\n  SPRT flagged controllers: {sorted(flagged) or 'none'}")


# ── Main entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EVBTCST Simulation")
    parser.add_argument("--quick",       action="store_true", help="30-trial smoke test")
    parser.add_argument("--tables-only", action="store_true", help="Skip plots")
    parser.add_argument("--plots-only",  action="store_true", help="Only regenerate plots")
    parser.add_argument("--lemma1",      action="store_true", help="Only Lemma 1 check")
    parser.add_argument("--table2",      action="store_true", help="Only Table 2")
    parser.add_argument("--table3",      action="store_true", help="Only Table 3")
    parser.add_argument("--table4",      action="store_true", help="Only Table 4")
    args = parser.parse_args()

    N          = 30  if args.quick else 500
    N_sweep    = 20  if args.quick else 150
    N_t2_batches = 30  if args.quick else 200   # parallel over k-values — fast
    N_t3       = 30  if args.quick else 300
    N_t4       = 20  if args.quick else 150

    # ── Lemma 1 only ──────────────────────────────────────────────────────
    if args.lemma1:
        run_lemma1()
        return

    # ── Table-only shortcuts ──────────────────────────────────────────────
    if args.table2:
        print("\nTable 2 — Detection Power vs k")
        df = run_table2(n_batches=N_t2_batches, out_dir=str(OUT), verbose=True)
        print(df.to_string(index=False))
        return

    if args.table3:
        print("\nTable 3 — Fidelity vs Byzantine Count")
        df = run_table3(n_trials=N_t3, out_dir=str(OUT), verbose=True)
        print(df.to_string(index=False))
        return

    if args.table4:
        print("\nTable 4 — Noise Scaling")
        df = run_table4(n_trials=N_t4, out_dir=str(OUT), verbose=True)
        print(df.to_string(index=False))
        return

    # ── Plots-only: load saved CSVs ───────────────────────────────────────
    if args.plots_only:
        def _load(name):
            p = OUT / name
            return pd.read_csv(p) if p.exists() else None

        generate_all_plots(
            df_default=_load("default_results.csv"),
            df_byz=_load("sweep_byzantine.csv"),
            df_noise=_load("sweep_noise.csv"),
            df_t2=_load("table2_detection_power.csv"),
            df_t3=_load("table3_fidelity_byzantine.csv"),
            df_t4=_load("table4_noise_scaling.csv"),
            out_dir=OUT,
        )
        log.info("Plots regenerated.")
        return

    # ── Full run ──────────────────────────────────────────────────────────
    log.info("Starting EVBTCST full simulation")

    # 1. Lemma 1 (fast, always run)
    run_lemma1()

    # 2. Default run
    df_default, sprt_default = run_default(n_trials=N)
    print_full_summary(df_default, sprt_default, label="default")

    # 3. Sweeps for figs 1-5
    df_byz   = run_sweep_byzantine(n_trials=N_sweep)
    df_noise = run_sweep_noise(n_trials=N_sweep)

    # 4. Table experiments
    print("\nTable 2 — Detection Power vs k")
    df_t2 = run_table2(n_batches=N_t2_batches, out_dir=str(OUT), verbose=True)

    print("\nTable 3 — Fidelity vs Byzantine Count")
    df_t3 = run_table3(n_trials=N_t3, out_dir=str(OUT), verbose=True)

    print("\nTable 4 — Noise Scaling")
    df_t4 = run_table4(n_trials=N_t4, out_dir=str(OUT), verbose=True)

    # 5. Plots
    if not args.tables_only:
        generate_all_plots(
            df_default=df_default,
            df_byz=df_byz,
            df_noise=df_noise,
            df_t2=df_t2,
            df_t3=df_t3,
            df_t4=df_t4,
            out_dir=OUT,
        )

    log.info("EVBTCST simulation complete.")


if __name__ == "__main__":
    # Required on Windows for ProcessPoolExecutor (spawn context)
    import multiprocessing
    multiprocessing.freeze_support()
    main()
