---

## 2) plots.py (komplett, inkl. optional Save)

```python
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from gewebe_reservoir import simulate_plv_modulated_trajectory


def plot_multi_seed_results(
    all_y: np.ndarray,
    all_phi: np.ndarray,
    all_plv: np.ndarray,
    all_corr: np.ndarray,
    n_steps: Optional[int] = None,
    base_gain: float = 0.15,
    dt: float = 0.01,
    min_plv_thr: float = 0.4,
    title: str = "Multi-seed PilotWeave-RC Results",
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualizes multi-seed results:
      - Mean ± std of rolling Pearson correlation
      - Mean ± std of PLV
      - Spaghetti + mean trajectory (PLV-gated)
    """
    if n_steps is None:
        n_steps = int(all_y.shape[1])
    t = np.arange(n_steps)

    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(3, 2, wspace=0.30, hspace=0.40)

    # 1) Rolling correlation (mean ± std)
    ax1 = fig.add_subplot(grid[0, 0])
    mean_corr = np.nanmean(all_corr, axis=0)
    std_corr = np.nanstd(all_corr, axis=0)
    ax1.plot(t, mean_corr, label="Mean corr(y, φ)", lw=2.2)
    ax1.fill_between(t, mean_corr - std_corr, mean_corr + std_corr, alpha=0.18)
    ax1.set_title("Statistical Coupling (rolling Pearson)")
    ax1.set_ylabel("Correlation")
    ax1.set_ylim(-0.5, 1.1)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right")

    # 2) PLV (mean ± std)
    ax2 = fig.add_subplot(grid[0, 1])
    mean_plv = np.nanmean(all_plv, axis=0)
    std_plv = np.nanstd(all_plv, axis=0)
    ax2.plot(t, mean_plv, label="Mean PLV", lw=2.2)
    ax2.fill_between(t, mean_plv - std_plv, mean_plv + std_plv, alpha=0.18)
    ax2.axhline(0.80, ls="--", alpha=0.7, label="High coherence threshold")
    ax2.set_title("Phase Synchronization (rolling PLV)")
    ax2.set_ylabel("PLV")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower right")

    # 3) Trajectories (spaghetti + mean)
    ax3 = fig.add_subplot(grid[1:, :])
    pos_list = []
    for s in range(all_y.shape[0]):
        pos = simulate_plv_modulated_trajectory(
            y=all_y[s],
            phi=all_phi[s],
            plv=all_plv[s],
            base_gain=base_gain,
            dt=dt,
            min_plv_thr=min_plv_thr,
        )
        pos_list.append(pos)
        ax3.plot(pos[:, 0], pos[:, 1], alpha=0.35, lw=1.0)

    pos_list = np.asarray(pos_list, dtype=np.float64)
    mean_pos = np.mean(pos_list, axis=0)

    ax3.plot(mean_pos[:, 0], mean_pos[:, 1], lw=3.0, label="Mean trajectory")
    ax3.plot(mean_pos[0, 0], mean_pos[0, 1], "o", ms=9, label="Start")
    ax3.plot(mean_pos[-1, 0], mean_pos[-1, 1], "x", ms=10, label="End")

    ax3.set_title(f"PLV-gated Trajectories (gain={base_gain}, min PLV thresh={min_plv_thr})")
    ax3.set_xlabel("X position (arbitrary units)")
    ax3.set_ylabel("Y position (arbitrary units)")
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)
    ax3.axis("equal")
    ax3.legend(loc="best")

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def print_ablation_summary(summary: Dict[str, Dict[str, float]]) -> None:
    """
    Pretty-print tail statistics from ablation study.
    """
    order = ["baseline", "shuffle_phi", "dm0", "boost0", "phi_only"]
    print("\n" + "=" * 66)
    print("Ablation tail summary (last tail_len steps, mean ± std over seeds)")
    print("-" * 66)
    for cond in order:
        s = summary.get(cond, {})
        plv_m = s.get("plv_tail_mean", np.nan)
        plv_s = s.get("plv_tail_std", np.nan)
        corr_m = s.get("corr_tail_mean", np.nan)
        corr_s = s.get("corr_tail_std", np.nan)
        print(
            f"{cond:>12} | PLV: {plv_m:>7.4f} ± {plv_s:<7.4f} | "
            f"corr: {corr_m:>7.4f} ± {corr_s:<7.4f}"
        )
    print("=" * 66 + "\n")


def plot_ablation_overview(
    results: Dict[str, Dict[str, np.ndarray]],
    summary: Dict[str, Dict[str, float]],
    n_steps: Optional[int] = None,
    title: str = "Ablation Study Overview (multi-seed)",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots mean ± std of corr and PLV for all ablation conditions.
    """
    conditions = ["baseline", "shuffle_phi", "dm0", "boost0", "phi_only"]
    any_cond = conditions[0]
    if n_steps is None:
        n_steps = int(results[any_cond]["plv"].shape[1])
    t = np.arange(n_steps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Correlation
    for cond in conditions:
        corr = results[cond]["corr"]
        m = np.nanmean(corr, axis=0)
        s = np.nanstd(corr, axis=0)
        ax1.plot(
            t,
            m,
            lw=2.0,
            label=f"{cond} (tail {summary[cond]['corr_tail_mean']:.3f} ± {summary[cond]['corr_tail_std']:.3f})",
        )
        ax1.fill_between(t, m - s, m + s, alpha=0.12)

    ax1.set_title("Rolling Pearson correlation (mean ± std)")
    ax1.set_ylabel("Correlation")
    ax1.set_ylim(-0.5, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=3, fontsize=9)

    # PLV
    for cond in conditions:
        plv = results[cond]["plv"]
        m = np.nanmean(plv, axis=0)
        s = np.nanstd(plv, axis=0)
        ax2.plot(
            t,
            m,
            lw=2.0,
            label=f"{cond} (tail {summary[cond]['plv_tail_mean']:.3f} ± {summary[cond]['plv_tail_std']:.3f})",
        )
        ax2.fill_between(t, m - s, m + s, alpha=0.12)

    ax2.axhline(0.80, ls="--", alpha=0.7, label="High coherence threshold")
    ax2.set_title("Rolling PLV (mean ± std)")
    ax2.set_ylabel("PLV")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time steps")
    ax2.legend(ncol=3, fontsize=9)

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

    # Print summary table in console
    print_ablation_summary(summary)
