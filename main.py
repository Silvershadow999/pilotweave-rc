from __future__ import annotations

import argparse

from gewebe_reservoir import ReservoirConfig, run_multi_seed, run_ablation_multiseed
from plots import plot_multi_seed_results, plot_ablation_overview, print_ablation_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pilotweave-rc",
        description="PilotWeave-RC: SciPy-free physical reservoir computing simulation using cosmic ray pulses and fuzzy DM surrogate waves.",
    )
    p.add_argument("--version", action="version", version="pilotweave-rc 0.1.0")

    sub = p.add_subparsers(dest="cmd", required=True, help="Available commands")

    def add_shared_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--n-steps", type=int, default=3300, help="Number of simulation steps")
        sp.add_argument("--fs", type=float, default=1000.0, help="Sampling frequency (Hz)")
        sp.add_argument("--window", type=int, default=400, help="Rolling window size for corr/PLV")
        sp.add_argument("--p-pulse", type=float, default=0.04, help="Probability of cosmic ray pulse")
        sp.add_argument("--pulse-sigma", type=float, default=0.12, help="Std dev of pulse amplitude")
        sp.add_argument("--no-persistent-noise", action="store_true", help="Disable persistent background noise")
        sp.add_argument("--no-detrend-plv", action="store_true", help="Disable detrending for PLV calculation")
        sp.add_argument("--seed-base", type=int, default=1234, help="Base seed for multi-seed runs")

        # Plot output (headless-friendly)
        sp.add_argument(
            "--save-plot",
            type=str,
            default="",
            help="If set, save plot to this PNG path instead of opening a window.",
        )

        # Diagnostics / curvature
        sp.add_argument(
            "--log-density",
            action="store_true",
            help="Log density/curvature/substeps and modulate trajectories with curvature.",
        )
        sp.add_argument(
            "--fixed-substeps",
            type=int,
            default=1,
            help="Force a fixed number of substeps per tick (>=1). Use 1 for density-based/normal.",
        )

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

        # density/curvature knobs (optional)
        sp.add_argument("--base-density", type=float, default=1.0)
        sp.add_argument("--pulse-density-boost", type=float, default=5.0)
        sp.add_argument("--activity-density-gain", type=float, default=0.10)
        sp.add_argument("--curvature-gain", type=float, default=2.0)
        sp.add_argument("--fr-power", type=float, default=0.37)
        sp.add_argument("--fr-max", type=float, default=6.0)

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
        seed=42,  # overwritten per seed
        dtype=args.dtype,
        # density/curvature
        log_density=bool(args.log_density),
        fixed_substeps=int(max(1, args.fixed_substeps)),
        base_density=float(args.base_density),
        pulse_density_boost=float(args.pulse_density_boost),
        activity_density_gain=float(args.activity_density_gain),
        curvature_gain=float(args.curvature_gain),
        fr_power=float(args.fr_power),
        fr_max=float(args.fr_max),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = cfg_from_args(args)
    persistent_noise = not args.no_persistent_noise
    detrend_plv = not args.no_detrend_plv
    save_path = args.save_plot.strip() or None

    if args.cmd == "multi-seed":
        all_y, all_phi, all_plv, all_corr, all_curv = run_multi_seed(
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
            all_curvature=all_curv if cfg.log_density else None,
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

        print_ablation_summary(summary)

    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
