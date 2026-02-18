from __future__ import annotations

import numpy as np

from gewebe_reservoir import ReservoirConfig, run_multi_seed, run_ablation_multiseed
from plots import plot_multi_seed_results, plot_ablation_overview, print_ablation_summary


def main() -> None:
    # ---- Baseline config (matches defaults) ----
    cfg = ReservoirConfig(
        n_nodes=300,
        dm_coupling=0.25,
        rho=0.92,
        eta=0.008,
        beta=0.0025,
        lambda_forget=8e-5,
        leak=0.25,
        sparsity_thresh=0.085,
        mem_clip=1.5,
        hebb_baseline=0.10,
        hebb_pulse_boost=6.0,
        persistent_noise_sigma=0.012,
        dm_boost_steps=1200,
        dm_boost_factor=2.0,
        dtype="float64",
        seed=42,
    )

    # ---- Demo parameters (shorter run so people can try quickly) ----
    n_steps = 2000
    fs = 1000.0
    window = 400
    p_pulse = 0.04
    pulse_sigma = 0.12

    # ============================================================
    # 1) Baseline multi-seed demo
    # ============================================================
    print("\n[1/2] Running baseline multi-seed demo...")
    all_y, all_phi, all_plv, all_corr = run_multi_seed(
        base_cfg=cfg,
        n_seeds=5,
        seed_base=1234,
        n_steps=n_steps,
        fs=fs,
        window=window,
        p_pulse=p_pulse,
        pulse_sigma=pulse_sigma,
        persistent_noise=True,
        detrend_plv=True,
    )

    plot_multi_seed_results(
        all_y=all_y,
        all_phi=all_phi,
        all_plv=all_plv,
        all_corr=all_corr,
        n_steps=n_steps,
        base_gain=0.15,
        dt=0.01,
        min_plv_thr=0.4,
        title="PilotWeave-RC Demo: baseline multi-seed (5 runs)",
    )

    # ============================================================
    # 2) Ablation demo (smaller seeds so it runs fast)
    # ============================================================
    print("\n[2/2] Running ablation demo (fast settings)...")
    results, summary = run_ablation_multiseed(
        base_cfg=cfg,
        n_seeds=6,
        seed_base=1234,
        n_steps=n_steps,
        fs=fs,
        window=window,
        p_pulse=p_pulse,
        pulse_sigma=pulse_sigma,
        persistent_noise=True,
        detrend_plv=True,
        tail_len=400,
    )

    plot_ablation_overview(
        results=results,
        summary=summary,
        n_steps=n_steps,
        title="PilotWeave-RC Demo: ablation overview (6 seeds/condition)",
    )

    print_ablation_summary(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
