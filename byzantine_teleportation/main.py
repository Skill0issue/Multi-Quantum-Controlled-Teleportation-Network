"""
main.py — Entry point, experiment runner, parameter sweeps.

Valid (n, f) pairs satisfying both constraints (n even, n > 3f):
  (4, 1), (8, 2), (12, 3), (16, 4)

Default: n=8, f=2, noise=0.01, 500 trials.
"""

import logging
import pandas as pd
from pathlib import Path

from config import SimConfig
from simulator import ByzantineQTSimulator
from analysis import compute_statistics, print_summary
from plotting import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("simulation.log")],
)
log = logging.getLogger("main")

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def run_default() -> pd.DataFrame:
    log.info("=== Default: n=8, f=2, noise=0.01, 5 trials ===")
    cfg = SimConfig(n_controllers=8, max_byzantine=2, noise_level=0.01, n_trials=5)
    df  = ByzantineQTSimulator(cfg).run()
    df.to_csv(OUT / "default_results.csv", index=False)
    return df


def sweep_byzantine(n_trials: int = 20) -> pd.DataFrame:
    """f = 0, 1, 2 with n=8."""
    frames = []
    for f in [0, 1, 2]:
        cfg = SimConfig(n_controllers=8, max_byzantine=f,
                        noise_level=0.01, n_trials=n_trials)
        df  = ByzantineQTSimulator(cfg).run()
        df["sweep_f"] = f
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "sweep_byzantine.csv", index=False)
    return result


def sweep_noise(n_trials: int = 20) -> pd.DataFrame:
    """noise ∈ {0, 0.005, 0.01, 0.02} with n=8, f=2."""
    frames = []
    for noise in [0.0, 0.005, 0.01, 0.02]:
        cfg = SimConfig(n_controllers=8, max_byzantine=2,
                        noise_level=noise, n_trials=n_trials)
        df  = ByzantineQTSimulator(cfg).run()
        df["sweep_noise"] = noise
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "sweep_noise.csv", index=False)
    return result


if __name__ == "__main__":
    log.info("Starting Byzantine-Tolerant QT Simulation")
    df_default = run_default()
    print_summary(compute_statistics(df_default))

    N = 20
    df_byz   = sweep_byzantine(N)
    df_noise = sweep_noise(N)

    generate_all_plots(df_default, df_byz, df_noise, OUT)
    log.info("Complete.")
