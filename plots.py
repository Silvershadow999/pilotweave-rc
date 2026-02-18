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
    all_curvature: Optional[np.ndarray] = None,
    all_rho: Optional[np.ndarray] = None,
    all_fr: Optional[np.ndarray] = None,
) -> None:
    """
    Visualizes multi-seed results:
      - Mean ± std of rolling Pearson correlation
      - Mean ± std of PLV
      - Spaghetti + mean trajectory (PLV-gated)
    If curvature logs are provided, trajectory uses curvature[t] multiplier
    and an extra diagnostics panel can be shown (rho/curv/fr).
    """
    if n_steps is None:
        n_steps = all_y.shape[1]
    t = np.arange(n_steps)

    want_diag = (all_curvature is not None) or (all_rho is not None) or (all_fr is not None)
    if want_diag:
        fig = plt.figure(figsize=(15, 13))
        grid = plt.GridSpec(4, 2, wspace=0.30, hspace=0.40)
    else:
        fig = plt.figure(figsize=figsize)
        grid = plt.GridSpec(3, 2, wspace=0.30, hspace=0.40)

    # 1) Rolling correlation (mean ± std)
    ax1 = fig.add_subplot(grid[0, 0])
    mean_corr = np.nanmean(all_corr, axis=0)
    std_corr = np.nanstd(all_corr, axis=0)
    ax1.plot(t, mean_corr, label="Mean corr(y, φ)", lw=2.2, color="#1f77b4")
    ax1.fill_between(t, mean_corr - std_corr, mean_corr + std_corr, alpha=0.18, color="#1f77b4")
    ax1.set_title("Statistical Coupling (rolling Pearson)")
    ax1.set_ylabel("Correlation")
    ax1.set_ylim(-0.5, 1.1)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right")

    # 2) PLV (mean ± std)
    ax2 = fig.add_subplot(grid[0, 1])
    mean_plv = np.nanmean(all_plv, axis=0)
    std_plv = np.nanstd(all_plv, axis=0)
    ax2.plot(t, mean_plv, label="Mean PLV", lw=2.2, color="#2ca02c")
    ax2.fill_between(t, mean_plv - std_plv, mean_plv + std_plv, alpha=0.18, color="#2ca02c")
    ax2.axhline(0.80, ls="--", color="orange", alpha=0.7, label="High coherence threshold")
    ax2.set_title("Phase Synchronization (rolling PLV)")
    ax2.set_ylabel("PLV")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower right")

    # 3) Trajectories (spaghetti + mean)
    ax3 = fig.add_subplot(grid[1:3, :])
    pos_list = []
    for s in range(all_y.shape[0]):
        curv = None
        if all_curvature is not None:
            curv = all_curvature[s]

        pos = simulate_plv_modulated_trajectory(
            y=all_y[s],
            phi=all_phi[s],
            plv=all_plv[s],
            base_gain=base_gain,
            dt=dt,
            min_plv_thr=min_plv_thr,
            curvature=curv,
        )
        pos_list.append(pos)
        ax3.plot(pos[:, 0], pos[:, 1], alpha=0.35, lw=1.0, color="gray")

    pos_list = np.array(pos_list)
    mean_pos = np.mean(pos_list, axis=0)

    ax3.plot(mean_pos[:, 0], mean_pos[:, 1], lw=3.5, color="#d62728", label="Mean trajectory")
    ax3.plot(mean_pos[0, 0], mean_pos[0, 1], "o", ms=10, color="green", label="Start")
    ax3.plot(mean_pos[-1, 0], mean_pos[-1, 1], "x", ms=12, color="purple", label="End")

    extra = " + curvature" if all_curvature is not None else ""
    ax3.set_title(f"PLV-gated Trajectories{extra} (gain={base_gain}, min PLV={min_plv_thr})")
    ax3.set_xlabel("X position (arb. units)")
    ax3.set_ylabel("Y position (arb. units)")
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)
    ax3.axis("equal")
    ax3.legend(loc="best")

    # 4) Optional diagnostics panel
    if want_diag:
        ax4 = fig.add_subplot(grid[3, 0])
        ax5 = fig.add_subplot(grid[3, 1])

        if all_rho is not None:
            m = np.nanmean(all_rho, axis=0)
            s = np.nanstd(all_rho, axis=0)
            ax4.plot(t, m, lw=2.0, label="ρ mean")
            ax4.fill_between(t, m - s, m + s, alpha=0.15)
        ax4.set_title("Local Density ρ(t) (mean ± std)")
        ax4.set_xlabel("Time steps")
        ax4.grid(True, alpha=0.25)
        ax4.legend(loc="best")

        if all_curvature is not None:
            m = np.nanmean(all_curvature, axis=0)
            s = np.nanstd(all_curvature, axis=0)
            ax5.plot(t, m, lw=2.0, label="curvature mean")
            ax5.fill_between(t, m - s, m + s, alpha=0.15)
        if all_fr is not None:
            m = np.nanmean(all_fr, axis=0)
            ax5.plot(t, m, lw=2.0, ls="--", label="substeps mean")
        ax5.set_title("Curvature & Substeps (mean)")
        ax5.set_xlabel("Time steps")
        ax5.grid(True, alpha=0.25)
        ax5.legend(loc="best")

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def print_ablation_summary(summary: Dict[str, Dict[str, float]]) -> None:
    order = ["baseline", "shuffle_phi", "dm0", "boost0", "phi_only"]
    print("\n" + "=" * 60)
    print("Ablation tail summary (last 600 steps, mean ± std over seeds)")
    print("-" * 60)
    for cond in order:
        s = summary.get(cond, {})
        plv_m = s.get("plv_tail_mean", np.nan)
        plv_s = s.get("plv_tail_std", np.nan)
        corr_m = s.get("corr_tail_mean", np.nan)
        corr_s = s.get("corr_tail_std", np.nan)
        print(f"{cond:>12} | PLV: {plv_m:>6.4f} ± {plv_s:<6.4f} | corr: {corr_m:>6.4f} ± {corr_s:<6.4f}")
    print("=" * 60 + "\n")


def plot_ablation_overview(
    results: Dict[str, Dict[str, np.ndarray]],
    summary: Dict[str, Dict[str, float]],
    n_steps: Optional[int] = None,
    title: str = "Ablation Study Overview (multi-seed)",
) -> None:
    conditions = ["baseline", "shuffle_phi", "dm0", "boost0", "phi_only"]
    any_cond = conditions[0]
    if n_steps is None:
        n_steps = results[any_cond]["plv"].shape[1]
    t = np.arange(n_steps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Correlation
    for cond in conditions:
        corr = results[cond]["corr"]
        m = np.nanmean(corr, axis=0)
        s = np.nanstd(corr, axis=0)
        ax1.plot(t, m, lw=2.0, label=f"{cond} (tail {summary[cond]['corr_tail_mean']:.3f} ± {summary[cond]['corr_tail_std']:.3f})")
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
        ax2.plot(t, m, lw=2.0, label=f"{cond} (tail {summary[cond]['plv_tail_mean']:.3f} ± {summary[cond]['plv_tail_std']:.3f})")
        ax2.fill_between(t, m - s, m + s, alpha=0.12)

    ax2.axhline(0.80, ls="--", color="orange", alpha=0.7, label="High coherence threshold")
    ax2.set_title("Rolling PLV (mean ± std)")
    ax2.set_ylabel("PLV")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time steps")
    ax2.legend(ncol=3, fontsize=9)

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

    print_ablation_summary(summary)    ax1 = fig.add_subplot(grid[0, 0])
    mean_corr = np.nanmean(all_corr, axis=0)
    std_corr = np.nanstd(all_corr, axis=0)
    ax1.plot(t, mean_corr, label="Mean corr(y, φ)", lw=2.2, color="#1f77b4")
    ax1.fill_between(t, mean_corr - std_corr, mean_corr + std_corr, alpha=0.18, color="#1f77b4")
    ax1.set_title("Statistical Coupling (rolling Pearson)")
    ax1.set_ylabel("Correlation")
    ax1.set_ylim(-0.5, 1.1)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right")

    # 2) PLV (mean ± std)
    ax2 = fig.add_subplot(grid[0, 1])
    mean_plv = np.nanmean(all_plv, axis=0)
    std_plv = np.nanstd(all_plv, axis=0)
    ax2.plot(t, mean_plv, label="Mean PLV", lw=2.2, color="#2ca02c")
    ax2.fill_between(t, mean_plv - std_plv, mean_plv + std_plv, alpha=0.18, color="#2ca02c")
    ax2.axhline(0.80, ls="--", color="orange", alpha=0.7, label="High coherence threshold")
    ax2.set_title("Phase Synchronization (rolling PLV)")
    ax2.set_ylabel("PLV")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower right")

    # 3) Trajectories (spaghetti + mean + ±1σ envelope)
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
        ax3.plot(pos[:, 0], pos[:, 1], alpha=0.35, lw=1.0, color="gray")

    pos_list = np.array(pos_list)
    mean_pos = np.mean(pos_list, axis=0)
    std_pos_x = np.std(pos_list[:, :, 0], axis=0)
    std_pos_y = np.std(pos_list[:, :, 1], axis=0)

    ax3.plot(mean_pos[:, 0], mean_pos[:, 1], lw=3.5, color="#d62728", label="Mean trajectory")
    ax3.fill_betweenx(mean_pos[:, 1],
                      mean_pos[:, 0] - std_pos_x,
                      mean_pos[:, 0] + std_pos_x,
                      alpha=0.12, color="#d62728", label="±1σ (x)")
    ax3.plot(mean_pos[0, 0], mean_pos[0, 1], "o", ms=10, color="green", label="Start")
    ax3.plot(mean_pos[-1, 0], mean_pos[-1, 1], "x", ms=12, color="purple", label="End")

    ax3.set_title(f"PLV-gated Trajectories (gain={base_gain}, min PLV thresh={min_plv_thr})")
    ax3.set_xlabel("X position (arbitrary units)")
    ax3.set_ylabel("Y position (arbitrary units)")
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)
    ax3.axis("equal")
    ax3.legend(loc="best")

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def print_ablation_summary(summary: Dict[str, Dict[str, float]]) -> None:
    """
    Pretty-print tail statistics from ablation study.
    """
    order = ["baseline", "shuffle_phi", "dm0", "boost0", "phi_only"]
    print("\n" + "="*60)
    print("Ablation tail summary (last 600 steps, mean ± std over seeds)")
    print("-"*60)
    for cond in order:
        s = summary.get(cond, {})
        plv_m = s.get("plv_tail_mean", np.nan)
        plv_s = s.get("plv_tail_std", np.nan)
        corr_m = s.get("corr_tail_mean", np.nan)
        corr_s = s.get("corr_tail_std", np.nan)
        print(f"{cond:>12} | PLV: {plv_m:>6.4f} ± {plv_s:<6.4f} | corr: {corr_m:>6.4f} ± {corr_s:<6.4f}")
    print("="*60 + "\n")


def plot_ablation_overview(
    results: Dict[str, Dict[str, np.ndarray]],
    summary: Dict[str, Dict[str, float]],
    n_steps: Optional[int] = None,
    title: str = "Ablation Study Overview (multi-seed)",
) -> None:
    """
    Plots mean ± std of corr and PLV for all ablation conditions.
    """
    conditions = ["baseline", "shuffle_phi", "dm0", "boost0", "phi_only"]
    any_cond = conditions[0]
    if n_steps is None:
        n_steps = results[any_cond]["plv"].shape[1]
    t = np.arange(n_steps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Correlation
    for cond in conditions:
        corr = results[cond]["corr"]
        m = np.nanmean(corr, axis=0)
        s = np.nanstd(corr, axis=0)
        ax1.plot(t, m, lw=2.0, label=f"{cond} (tail {summary[cond]['corr_tail_mean']:.3f} ± {summary[cond]['corr_tail_std']:.3f})")
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
        ax2.plot(t, m, lw=2.0, label=f"{cond} (tail {summary[cond]['plv_tail_mean']:.3f} ± {summary[cond]['plv_tail_std']:.3f})")
        ax2.fill_between(t, m - s, m + s, alpha=0.12)

    ax2.axhline(0.80, ls="--", color="orange", alpha=0.7, label="High coherence threshold")
    ax2.set_title("Rolling PLV (mean ± std)")
    ax2.set_ylabel("PLV")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time steps")
    ax2.legend(ncol=3, fontsize=9)

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

    # Print summary table in console
    print_ablation_summary(summary)
