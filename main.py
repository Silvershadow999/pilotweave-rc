# main.py
from __future__ import annotations

import argparse

from gewebe_reservoir import (
    ReservoirConfig,
    run_multi_seed,
    run_ablation_multiseed,
)
from plots import (
    plot_multi_seed_results,
    plot_ablation_overview,
    print_ablation_summary,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pilotweave-rc",
        description="PilotWeave-RC: SciPy-free physical reservoir computing simulation using cosmic ray pulses and fuzzy DM surrogate waves.",
    )
    p.add_argument("--version", action="version", version="pilotweave-rc 0.1.0")

    sub = p.add_subparsers(dest="cmd", required=True, help="Available commands")

    def add_shared_args(sp: argparse.ArgumentParser):
        sp.add_argument("--n-steps", type=int, default=3300, help="Number of simulation steps")
        sp.add_argument("--fs", type=float, default=1000.0, help="Sampling frequency (Hz)")
        sp.add_argument("--window", type=int, default=400, help="Rolling window size for corr/PLV")
        sp.add_argument("--p-pulse", type=float, default=0.04, help="Probability of cosmic ray pulse")
        sp.add_argument("--pulse-sigma", type=float, default=0.12, help="Std dev of pulse amplitude")
        sp.add_argument("--no-persistent-noise", action="store_true", help="Disable persistent background noise")
        sp.add_argument("--no-detrend-plv", action="store_true", help="Disable detrending for PLV calculation")
        sp.add_argument("--seed-base", type=int, default=1234, help="Base seed for multi-seed runs")

        # Model config overrides
        sp.add_argument("--n-nodes", type=int, default=300)
        sp.add_argument("--rho", type=float, default=0.92)
        sp.add_argument("--eta", type=float, default=0.008)
        sp.add_argument("--beta", type=float, default=0.0025)
        sp.add_argument("--lambda-forget", type=float, default=8e-5)
        sp.add_argument("--dm-coupling", type=float, default=0.25)
        sp.add_argument("--leak", type=float, default=0.25)
        sp.add_argument("--sparsity-thresh", type=float, default=0.085)
        sp.add_argument("--mem-clip", type=float, default=1.5)
        sp.add_argument("--hebb-baseline", type=float, default=0.10)
        sp.add_argument("--hebb-pulse-boost", type=float, default=6.0)
        sp.add_argument("--persistent-noise-sigma", type=float, default=0.012)
        sp.add_argument("--dm-boost-steps", type=int, default=1200)
        sp.add_argument("--dm-boost-factor", type=float, default=2.0)
        sp.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])

        # Headless save
        sp.add_argument(
            "--save-plot",
            type=str,
            default="",
            help="Save plot to PNG (headless). Example: out/multiseed.png",
        )

    # multi-seed
    ms = sub.add_parser("multi-seed", help="Run baseline multi-seed experiment and plot results.")
    add_shared_args(ms)
    ms.add_argument("--n-seeds", type=int, default=10, help="Number of independent runs (seeds)")
    ms.add_argument("--traj-gain", type=float, default=0.15, help="Base gain factor for PLV-modulated thrust")
    ms.add_argument("--traj-dt", type=float, default=0.01, help="Time step for trajectory integration")
    ms.add_argument("--traj-min-plv", type=float, default=0.4, help="Minimum PLV threshold for thrust application")

    # ablation
    ab = sub.add_parser("ablation", help="Run five-condition ablation multi-seed and show overview.")
    add_shared_args(ab)
    ab.add_argument("--n-seeds", type=int, default=20, help="Number of independent runs per condition")
    ab.add_argument("--tail-len", type=int, default=600, help="Number of tail steps for statistics")

    return p


def cfg_from_args(args: argparse.Namespace) -> ReservoirConfig:
    return ReservoirConfig(
        n_nodes=args.n_nodes,
        rho=args.rho,
        eta=args.eta,
        beta=args.beta,
        lambda_forget=args.lambda_forget,
        dm_coupling=args.dm_coupling,
        leak=args.leak,
        sparsity_thresh=args.sparsity_thresh,
        mem_clip=args.mem_clip,
        hebb_baseline=args.hebb_baseline,
        hebb_pulse_boost=args.hebb_pulse_boost,
        persistent_noise_sigma=args.persistent_noise_sigma,
        dm_boost_steps=args.dm_boost_steps,
        dm_boost_factor=args.dm_boost_factor,
        dtype=args.dtype,
        seed=42,  # overwritten per seed in multi-run functions
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = cfg_from_args(args)
    persistent_noise = not args.no_persistent_noise
    detrend_plv = not args.no_detrend_plv
    save_path = args.save_plot.strip() or None

    if args.cmd == "multi-seed":
        all_y, all_phi, all_plv, all_corr = run_multi_seed(
            base_cfg=cfg,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
            n_steps=args.n_steps,
            fs=args.fs,
            window=args.window,
            p_pulse=args.p_pulse,
            pulse_sigma=args.pulse_sigma,
            persistent_noise=persistent_noise,
            detrend_plv=detrend_plv,
        )

        plot_multi_seed_results(
            all_y=all_y,
            all_phi=all_phi,
            all_plv=all_plv,
            all_corr=all_corr,
            n_steps=args.n_steps,
            base_gain=args.traj_gain,
            dt=args.traj_dt,
            min_plv_thr=args.traj_min_plv,
            title=f"PilotWeave-RC multi-seed ({args.n_seeds} runs)",
            save_path=save_path,
        )

    elif args.cmd == "ablation":
        results, summary = run_ablation_multiseed(
            base_cfg=cfg,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
            n_steps=args.n_steps,
            fs=args.fs,
            window=args.window,
            p_pulse=args.p_pulse,
            pulse_sigma=args.pulse_sigma,
            persistent_noise=persistent_noise,
            detrend_plv=detrend_plv,
            tail_len=args.tail_len,
        )

        plot_ablation_overview(
            results=results,
            summary=summary,
            n_steps=args.n_steps,
            title=f"PilotWeave-RC ablation ({args.n_seeds} seeds / condition)",
            save_path=save_path,
        )

        # optional extra console summary (already printed inside plot_ablation_overview)
        print_ablation_summary(summary)

    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()        # Model config overrides
        sp.add_argument("--n-nodes", type=int, default=300)
        sp.add_argument("--rho", type=float, default=0.92)
        sp.add_argument("--eta", type=float, default=0.008)
        sp.add_argument("--beta", type=float, default=0.0025)
        sp.add_argument("--lambda-forget", type=float, default=8e-5)
        sp.add_argument("--dm-coupling", type=float, default=0.25)
        sp.add_argument("--leak", type=float, default=0.25)
        sp.add_argument("--sparsity-thresh", type=float, default=0.085)
        sp.add_argument("--mem-clip", type=float, default=1.5)
        sp.add_argument("--hebb-baseline", type=float, default=0.10)
        sp.add_argument("--hebb-pulse-boost", type=float, default=6.0)
        sp.add_argument("--persistent-noise-sigma", type=float, default=0.012)
        sp.add_argument("--dm-boost-steps", type=int, default=1200)
        sp.add_argument("--dm-boost-factor", type=float, default=2.0)
        sp.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])

        # Curvature/density controls (repo-fertig + fair testbar)
        sp.add_argument("--curvature", action="store_true", help="Enable density/curvature mode (simulation-only).")
        sp.add_argument("--log-density", action="store_true", help="Log rho/curvature/substeps for plotting diagnostics.")
        sp.add_argument("--fixed-substeps", type=int, default=1,
                        help="Compute-matched mode: force a fixed number of substeps per tick (>=1). Overrides density substeps.")
        sp.add_argument("--fr-max", type=float, default=6.0, help="Max substeps cap (runtime safety).")
        sp.add_argument("--curvature-gain", type=float, default=2.0, help="Curvature amplification (saturating).")
        sp.add_argument("--pulse-density-boost", type=float, default=5.0, help="Density boost when pulse is active.")
        sp.add_argument("--activity-density-gain", type=float, default=0.10, help="Density gain from reservoir activity.")
        sp.add_argument("--fr-power", type=float, default=0.37, help="Exponent for density->framerate mapping.")
        sp.add_argument("--base-density", type=float, default=1.0, help="Baseline density for normalization.")

    # multi-seed
    ms = sub.add_parser("multi-seed", help="Run multi-seed experiment and plot results.")
    add_shared_args(ms)
    ms.add_argument("--n-seeds", type=int, default=10, help="Number of independent runs (seeds)")

    # Trajectory params
    ms.add_argument("--traj-gain", type=float, default=0.15, help="Base gain factor for PLV-modulated thrust")
    ms.add_argument("--traj-dt", type=float, default=0.01, help="Time step for trajectory integration")
    ms.add_argument("--traj-min-plv", type=float, default=0.4, help="Minimum PLV threshold for thrust application")

    # ablation
    ab = sub.add_parser("ablation", help="Run five-condition ablation multi-seed and show overview.")
    add_shared_args(ab)
    ab.add_argument("--n-seeds", type=int, default=20, help="Number of independent runs per condition")
    ab.add_argument("--tail-len", type=int, default=600, help="Number of tail steps for statistics")

    return p


def cfg_from_args(args: argparse.Namespace) -> ReservoirConfig:
    return ReservoirConfig(
        n_nodes=args.n_nodes,
        rho=args.rho,
        eta=args.eta,
        beta=args.beta,
        lambda_forget=args.lambda_forget,
        dm_coupling=args.dm_coupling,
        leak=args.leak,
        sparsity_thresh=args.sparsity_thresh,
        mem_clip=args.mem_clip,
        hebb_baseline=args.hebb_baseline,
        hebb_pulse_boost=args.hebb_pulse_boost,
        persistent_noise_sigma=args.persistent_noise_sigma,
        dm_boost_steps=args.dm_boost_steps,
        dm_boost_factor=args.dm_boost_factor,
        dtype=args.dtype,
        seed=42,  # overwritten per seed in multi-run functions

        # curvature/density
        base_density=args.base_density,
        pulse_density_boost=args.pulse_density_boost,
        activity_density_gain=args.activity_density_gain,
        curvature_gain=args.curvature_gain,
        fr_power=args.fr_power,
        fr_max=args.fr_max,

        # fairness
        use_density_substeps=bool(args.curvature),
        fixed_substeps=int(args.fixed_substeps),
        log_density_default=bool(args.log_density),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = cfg_from_args(args)
    persistent_noise = not args.no_persistent_noise
    detrend_plv = not args.no_detrend_plv

    if args.cmd == "multi-seed":
        if args.curvature or args.log_density or args.fixed_substeps > 1:
            (all_y, all_phi, all_plv, all_corr, all_rho, all_curv, all_fr) = run_multi_seed_curvature(
                base_cfg=cfg,
                n_seeds=args.n_seeds,
                seed_base=args.seed_base,
                n_steps=args.n_steps,
                fs=args.fs,
                window=args.window,
                p_pulse=args.p_pulse,
                pulse_sigma=args.pulse_sigma,
                persistent_noise=persistent_noise,
                detrend_plv=detrend_plv,
                log_density=True,
            )

            plot_multi_seed_results(
                all_y=all_y,
                all_phi=all_phi,
                all_plv=all_plv,
                all_corr=all_corr,
                n_steps=args.n_steps,
                base_gain=args.traj_gain,
                dt=args.traj_dt,
                min_plv_thr=args.traj_min_plv,
                title=f"PilotWeave-RC multi-seed ({args.n_seeds} runs) | curvature={args.curvature} | fixed_substeps={args.fixed_substeps}",
                all_curvature=all_curv if args.curvature else None,
                all_rho=all_rho if args.log_density else None,
                all_fr=all_fr if args.log_density else None,
            )
        else:
            all_y, all_phi, all_plv, all_corr = run_multi_seed(
                base_cfg=cfg,
                n_seeds=args.n_seeds,
                seed_base=args.seed_base,
                n_steps=args.n_steps,
                fs=args.fs,
                window=args.window,
                p_pulse=args.p_pulse,
                pulse_sigma=args.pulse_sigma,
                persistent_noise=persistent_noise,
                detrend_plv=detrend_plv,
            )

            plot_multi_seed_results(
                all_y=all_y,
                all_phi=all_phi,
                all_plv=all_plv,
                all_corr=all_corr,
                n_steps=args.n_steps,
                base_gain=args.traj_gain,
                dt=args.traj_dt,
                min_plv_thr=args.traj_min_plv,
                title=f"PilotWeave-RC multi-seed ({args.n_seeds} runs)",
            )

    elif args.cmd == "ablation":
        results, summary = run_ablation_multiseed(
            base_cfg=cfg,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
            n_steps=args.n_steps,
            fs=args.fs,
            window=args.window,
            p_pulse=args.p_pulse,
            pulse_sigma=args.pulse_sigma,
            persistent_noise=persistent_noise,
            detrend_plv=detrend_plv,
            tail_len=args.tail_len,
        )

        plot_ablation_overview(
            results=results,
            summary=summary,
            n_steps=args.n_steps,
            title=f"PilotWeave-RC ablation ({args.n_seeds} seeds / condition)",
        )
        print_ablation_summary(summary)

    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()        sp.add_argument("--seed-base", type=int, default=1234, help="Base seed for multi-seed runs")

        # Model config overrides
        sp.add_argument("--n-nodes", type=int, default=300)
        sp.add_argument("--rho", type=float, default=0.92)
        sp.add_argument("--eta", type=float, default=0.008)
        sp.add_argument("--beta", type=float, default=0.0025)
        sp.add_argument("--lambda-forget", type=float, default=8e-5)
        sp.add_argument("--dm-coupling", type=float, default=0.25)
        sp.add_argument("--leak", type=float, default=0.25)
        sp.add_argument("--sparsity-thresh", type=float, default=0.085)
        sp.add_argument("--mem-clip", type=float, default=1.5)
        sp.add_argument("--hebb-baseline", type=float, default=0.10)
        sp.add_argument("--hebb-pulse-boost", type=float, default=6.0)
        sp.add_argument("--persistent-noise-sigma", type=float, default=0.012)
        sp.add_argument("--dm-boost-steps", type=int, default=1200)
        sp.add_argument("--dm-boost-factor", type=float, default=2.0)
        sp.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])

    # multi-seed
    ms = sub.add_parser("multi-seed", help="Run baseline multi-seed experiment and visualize results.")
    add_shared_args(ms)
    ms.add_argument("--n-seeds", type=int, default=10, help="Number of independent runs (seeds)")
    ms.add_argument("--traj-gain", type=float, default=0.15, help="Base gain for PLV-modulated thrust")
    ms.add_argument("--traj-dt", type=float, default=0.01, help="Time step for trajectory integration")
    ms.add_argument("--traj-min-plv", type=float, default=0.4, help="Minimum PLV threshold for thrust gate")

    # ablation
    ab = sub.add_parser("ablation", help="Run five-condition ablation (multi-seed) and show overview.")
    add_shared_args(ab)
    ab.add_argument("--n-seeds", type=int, default=20, help="Number of independent runs per condition")
    ab.add_argument("--tail-len", type=int, default=600, help="Tail length for summary statistics")

    return p


def cfg_from_args(args: argparse.Namespace) -> ReservoirConfig:
    return ReservoirConfig(
        n_nodes=args.n_nodes,
        rho=args.rho,
        eta=args.eta,
        beta=args.beta,
        lambda_forget=args.lambda_forget,
        dm_coupling=args.dm_coupling,
        leak=args.leak,
        sparsity_thresh=args.sparsity_thresh,
        mem_clip=args.mem_clip,
        hebb_baseline=args.hebb_baseline,
        hebb_pulse_boost=args.hebb_pulse_boost,
        persistent_noise_sigma=args.persistent_noise_sigma,
        dm_boost_steps=args.dm_boost_steps,
        dm_boost_factor=args.dm_boost_factor,
        dtype=args.dtype,
        seed=42,  # overwritten inside run_multi_seed / run_ablation_multiseed
    )


def main() -> None:
    args = build_parser().parse_args()

    cfg = cfg_from_args(args)
    persistent_noise = not args.no_persistent_noise
    detrend_plv = not args.no_detrend_plv

    if args.cmd == "multi-seed":
        all_y, all_phi, all_plv, all_corr = run_multi_seed(
            base_cfg=cfg,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
            n_steps=args.n_steps,
            fs=args.fs,
            window=args.window,
            p_pulse=args.p_pulse,
            pulse_sigma=args.pulse_sigma,
            persistent_noise=persistent_noise,
            detrend_plv=detrend_plv,
        )
        plot_multi_seed_results(
            all_y=all_y,
            all_phi=all_phi,
            all_plv=all_plv,
            all_corr=all_corr,
            n_steps=args.n_steps,
            base_gain=args.traj_gain,
            dt=args.traj_dt,
            min_plv_thr=args.traj_min_plv,
            title=f"PilotWeave-RC multi-seed ({args.n_seeds} runs)",
        )

    elif args.cmd == "ablation":
        results, summary = run_ablation_multiseed(
            base_cfg=cfg,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
            n_steps=args.n_steps,
            fs=args.fs,
            window=args.window,
            p_pulse=args.p_pulse,
            pulse_sigma=args.pulse_sigma,
            persistent_noise=persistent_noise,
            detrend_plv=detrend_plv,
            tail_len=args.tail_len,
        )
        plot_ablation_overview(
            results=results,
            summary=summary,
            n_steps=args.n_steps,
            title=f"PilotWeave-RC ablation ({args.n_seeds} seeds / condition)",
        )
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
