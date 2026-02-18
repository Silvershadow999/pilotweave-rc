from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

import numpy as np


# ============================================================
# Utilities: spectral radius, rolling corr, Hilbert/PLV
# ============================================================
def power_iteration_spectral_radius(
    W: np.ndarray,
    n_iter: int = 60,
    rng: Optional[np.random.Generator] = None
) -> float:
    """Approximate spectral radius via power iteration."""
    if rng is None:
        rng = np.random.default_rng(0)
    v = rng.normal(size=W.shape[0])
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(n_iter):
        v = W @ v
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            return 0.0
        v /= nv
    wv = W @ v
    lam = float(np.dot(v, wv))
    return float(abs(lam))


def rolling_corr_fast(a: np.ndarray, b: np.ndarray, win: int) -> np.ndarray:
    """
    Rolling Pearson correlation (O(n)) via cumulative sums.
    Returns NaNs for indices < win-1.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.size
    out = np.full(n, np.nan, dtype=np.float64)
    if win <= 1 or win > n:
        return out

    ca = np.cumsum(a)
    cb = np.cumsum(b)
    ca2 = np.cumsum(a * a)
    cb2 = np.cumsum(b * b)
    cab = np.cumsum(a * b)

    s_a = ca[win - 1:] - np.concatenate(([0.0], ca[:-win]))
    s_b = cb[win - 1:] - np.concatenate(([0.0], cb[:-win]))
    s_a2 = ca2[win - 1:] - np.concatenate(([0.0], ca2[:-win]))
    s_b2 = cb2[win - 1:] - np.concatenate(([0.0], cb2[:-win]))
    s_ab = cab[win - 1:] - np.concatenate(([0.0], cab[:-win]))

    mean_a = s_a / win
    mean_b = s_b / win
    var_a = s_a2 / win - mean_a * mean_a
    var_b = s_b2 / win - mean_b * mean_b
    cov_ab = s_ab / win - mean_a * mean_b

    denom = np.sqrt(np.maximum(var_a, 0.0) * np.maximum(var_b, 0.0))
    good = denom > 1e-14
    corr = np.full_like(denom, np.nan)
    corr[good] = cov_ab[good] / denom[good]

    out[win - 1:] = corr
    return out


def hilbert_analytic_signal(x: np.ndarray, detrend: bool = True) -> np.ndarray:
    """
    Minimal Hilbert transform via FFT -> analytic signal, SciPy-free.
    Optionally remove mean to stabilize phase unwrap.
    """
    x = np.asarray(x, dtype=np.float64)
    if detrend:
        x = x - np.mean(x)

    n = x.size
    Xf = np.fft.fft(x)

    H = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        H[0] = 1.0
        H[n // 2] = 1.0
        H[1:n // 2] = 2.0
    else:
        H[0] = 1.0
        H[1:(n + 1) // 2] = 2.0

    return np.fft.ifft(Xf * H)


def rolling_plv(y: np.ndarray, phi: np.ndarray, win: int, detrend: bool = True) -> np.ndarray:
    """
    Rolling Phase Locking Value:
      PLV(t) = | mean_{window} exp(i*(phase_y - phase_phi)) |
    Returns NaNs for indices < win-1.
    """
    y = np.asarray(y, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    n = len(y)

    out = np.full(n, np.nan, dtype=np.float64)
    if win <= 1 or win > n:
        return out

    ay = hilbert_analytic_signal(y, detrend=detrend)
    ap = hilbert_analytic_signal(phi, detrend=detrend)
    phase_y = np.unwrap(np.angle(ay))
    phase_p = np.unwrap(np.angle(ap))

    dphi = phase_y - phase_p
    z = np.exp(1j * dphi)

    zr = np.real(z)
    zi = np.imag(z)
    cr = np.cumsum(zr)
    ci = np.cumsum(zi)

    sr = cr[win - 1:] - np.concatenate(([0.0], cr[:-win]))
    si = ci[win - 1:] - np.concatenate(([0.0], ci[:-win]))

    mr = sr / win
    mi = si / win
    out[win - 1:] = np.sqrt(mr * mr + mi * mi)
    return out


# ============================================================
# Config + Reservoir
# ============================================================
@dataclass
class ReservoirConfig:
    n_nodes: int = 300
    rho: float = 0.92
    eta: float = 0.008
    beta: float = 0.0025
    lambda_forget: float = 8e-5
    dm_coupling: float = 0.25
    leak: float = 0.25
    sparsity_thresh: float = 0.085
    mem_clip: float = 1.5

    # Hebbian update shaping
    hebb_baseline: float = 0.10
    hebb_pulse_boost: float = 6.0

    # Input noise
    persistent_noise_sigma: float = 0.012

    # DM boost phase
    dm_boost_steps: int = 1200
    dm_boost_factor: float = 2.0

    # RNG + dtype
    seed: int = 42
    dtype: str = "float64"  # use string for easy CLI

    def as_kwargs(self) -> Dict:
        d = asdict(self)
        d["dtype"] = np.float64 if self.dtype == "float64" else np.float32
        return d


class AdaptiveGewebeReservoirV3:
    """
    Quasikristallines Reservoir + memristives Gedächtnis (W_mem) + DM-Surrogat.
    Output y(t) wird ZEITKONSISTENT berechnet: y(t) = W_mem(t)·x(t)
    """

    def __init__(self, cfg: ReservoirConfig):
        self.cfg = cfg
        kw = cfg.as_kwargs()
        self.n_nodes = kw["n_nodes"]
        self.rng = np.random.default_rng(kw["seed"])
        self.dtype = kw["dtype"]

        # Store params (fast access)
        self.rho = float(kw["rho"])
        self.eta = float(kw["eta"])
        self.beta = float(kw["beta"])
        self.lambda_forget = float(kw["lambda_forget"])
        self.dm_coupling = float(kw["dm_coupling"])
        self.leak = float(kw["leak"])
        self.sparsity_thresh = float(kw["sparsity_thresh"])
        self.mem_clip = float(kw["mem_clip"])
        self.hebb_baseline = float(kw["hebb_baseline"])
        self.hebb_pulse_boost = float(kw["hebb_pulse_boost"])
        self.persistent_noise_sigma = float(kw["persistent_noise_sigma"])
        self.dm_boost_steps = int(kw["dm_boost_steps"])
        self.dm_boost_factor = float(kw["dm_boost_factor"])

        # Quasikristalline Kopplung W (aperiodic, sparse)
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        i = np.arange(self.n_nodes)[:, None]
        j = np.arange(self.n_nodes)[None, :]
        dist = np.abs(i - j * phi) % 1.0
        mask = dist < self.sparsity_thresh
        np.fill_diagonal(mask, False)

        W = np.zeros((self.n_nodes, self.n_nodes), dtype=self.dtype)
        W[mask] = self.rng.normal(0.0, 1.0, size=int(mask.sum())).astype(self.dtype)
        W = 0.5 * (W + W.T)

        sr = power_iteration_spectral_radius(W, n_iter=60, rng=self.rng)
        if sr > 1e-12:
            W *= (self.rho / sr)
        self.W = W

        # States
        self.x = np.zeros(self.n_nodes, dtype=self.dtype)
        self.W_mem = np.zeros(self.n_nodes, dtype=self.dtype)

        # Histories
        self.X_hist: Optional[np.ndarray] = None
        self.phi_hist: Optional[np.ndarray] = None
        self.u_hist: Optional[np.ndarray] = None
        self.y_hist: Optional[np.ndarray] = None

    def get_fdm_field(self, t: int, fs: float = 1000.0) -> float:
        """
        FDM surrogate: sum of sinusoids + noise.
        Frequencies are in Hz; time step is 1/fs seconds.
        """
        tt = t / fs
        base = math.sin(2 * math.pi * 0.963 * tt)
        harm = 0.4 * math.sin(2 * math.pi * 2.47 * tt)
        slow = 0.15 * math.sin(2 * math.pi * 0.031 * tt)
        noise = 0.08 * float(self.rng.normal())
        return float(base + harm + slow + noise)

    def step(self, u_rad: float, phi_dm: float, t: int) -> None:
        feedback = self.beta * (self.W_mem * self.x)

        effective_alpha = self.dm_coupling
        if t < self.dm_boost_steps:
            effective_alpha *= self.dm_boost_factor

        syn = (self.W @ self.x) + u_rad + (effective_alpha * phi_dm) + feedback
        x_new = np.tanh(syn)
        self.x = (1.0 - self.leak) * self.x + self.leak * x_new

        update_strength = self.eta * (self.hebb_baseline + self.hebb_pulse_boost * abs(u_rad))
        self.W_mem += update_strength * (self.x * phi_dm)

        self.W_mem *= (1.0 - self.lambda_forget)
        np.clip(self.W_mem, -self.mem_clip, self.mem_clip, out=self.W_mem)

    def run(
        self,
        n_steps: int,
        p_pulse: float = 0.04,
        pulse_sigma: float = 0.12,
        fs: float = 1000.0,
        persistent_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (u_hist, y_hist, phi_hist).
        """
        self.X_hist = np.zeros((n_steps, self.n_nodes), dtype=self.dtype)
        self.phi_hist = np.zeros(n_steps, dtype=np.float64)
        self.u_hist = np.zeros(n_steps, dtype=np.float64)
        self.y_hist = np.zeros(n_steps, dtype=np.float64)

        for t in range(n_steps):
            phi_dm = self.get_fdm_field(t, fs=fs)
            u_rad = float(self.rng.normal(0.0, pulse_sigma)) if (float(self.rng.random()) < p_pulse) else 0.0
            if persistent_noise:
                u_rad += float(self.rng.normal(0.0, self.persistent_noise_sigma))

            self.step(u_rad, phi_dm, t)

            self.X_hist[t] = self.x
            self.phi_hist[t] = phi_dm
            self.u_hist[t] = u_rad
            self.y_hist[t] = float(np.dot(self.W_mem, self.x))

        return self.u_hist, self.y_hist, self.phi_hist


# ============================================================
# Trajectory (PLV-modulated propulsion)
# ============================================================
def simulate_plv_modulated_trajectory(
    y: np.ndarray,
    phi: np.ndarray,
    plv: np.ndarray,
    base_gain: float = 0.15,
    dt: float = 0.01,
    min_plv_thr: float = 0.4,
) -> np.ndarray:
    """
    2D guidance: thrust direction follows DM phase; thrust magnitude scales with PLV gate.
    """
    y = np.asarray(y, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    plv = np.asarray(plv, dtype=np.float64)
    n = len(y)

    pos = np.zeros((n, 2), dtype=np.float64)
    vel = np.zeros((n, 2), dtype=np.float64)
    vel[0] = [1.0, 0.0]

    phase_phi = np.unwrap(np.angle(hilbert_analytic_signal(phi, detrend=True)))

    for t in range(1, n):
        p = plv[t]
        if not np.isfinite(p):
            p = 0.0
        g = base_gain * max(0.0, p - min_plv_thr) / (1.0 - min_plv_thr + 1e-12)
        thrust_dir = np.array([math.cos(phase_phi[t]), math.sin(phase_phi[t])], dtype=np.float64)
        thrust = g * y[t] * thrust_dir
        vel[t] = vel[t - 1] + thrust * dt
        pos[t] = pos[t - 1] + vel[t] * dt

    return pos


# ============================================================
# Experiment runners (multi-seed + ablation)
# ============================================================
def run_multi_seed(
    base_cfg: ReservoirConfig,
    n_seeds: int = 10,
    seed_base: int = 1234,
    n_steps: int = 3300,
    fs: float = 1000.0,
    window: int = 400,
    p_pulse: float = 0.04,
    pulse_sigma: float = 0.12,
    persistent_noise: bool = True,
    detrend_plv: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_y, all_phi, all_plv, all_corr = [], [], [], []
    for s in range(n_seeds):
        cfg = ReservoirConfig(**{**asdict(base_cfg), "seed": seed_base + s, "dtype": base_cfg.dtype})
        sim = AdaptiveGewebeReservoirV3(cfg)
        _, y, phi = sim.run(n_steps, p_pulse=p_pulse, pulse_sigma=pulse_sigma, fs=fs, persistent_noise=persistent_noise)
        plv = rolling_plv(y, phi, win=window, detrend=detrend_plv)
        corr = rolling_corr_fast(y, phi, win=window)

        all_y.append(y)
        all_phi.append(phi)
        all_plv.append(plv)
        all_corr.append(corr)

    return (np.asarray(all_y), np.asarray(all_phi), np.asarray(all_plv), np.asarray(all_corr))


def run_condition(
    base_cfg: ReservoirConfig,
    condition: str,
    seed: int,
    n_steps: int,
    fs: float,
    window: int,
    p_pulse: float,
    pulse_sigma: float,
    persistent_noise: bool,
    detrend_plv: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg_dict = asdict(base_cfg)
    cfg_dict["seed"] = int(seed)

    if condition == "dm0":
        cfg_dict["dm_coupling"] = 0.0
    elif condition == "boost0":
        cfg_dict["hebb_pulse_boost"] = 0.0

    cfg = ReservoirConfig(**cfg_dict)
    sim = AdaptiveGewebeReservoirV3(cfg)

    p = 0.0 if condition == "phi_only" else p_pulse
    _, y, phi = sim.run(n_steps, p_pulse=p, pulse_sigma=pulse_sigma, fs=fs, persistent_noise=persistent_noise)

    if condition == "shuffle_phi":
        rng = np.random.default_rng(seed + 999)
        rng.shuffle(phi)

    plv = rolling_plv(y, phi, win=window, detrend=detrend_plv)
    corr = rolling_corr_fast(y, phi, win=window)
    return y, phi, plv, corr


def run_ablation_multiseed(
    base_cfg: ReservoirConfig,
    n_seeds: int = 20,
    seed_base: int = 1234,
    n_steps: int = 3300,
    fs: float = 1000.0,
    window: int = 400,
    p_pulse: float = 0.04,
    pulse_sigma: float = 0.12,
    persistent_noise: bool = True,
    detrend_plv: bool = True,
    tail_len: int = 600,
) -> Tuple[Dict, Dict]:
    conditions = ["baseline", "shuffle_phi", "dm0", "boost0", "phi_only"]
    results = {c: {"y": [], "phi": [], "plv": [], "corr": []} for c in conditions}

    for s in range(n_seeds):
        seed = seed_base + s
        for cond in conditions:
            y, phi, plv, corr = run_condition(
                base_cfg, cond, seed, n_steps, fs, window, p_pulse, pulse_sigma, persistent_noise, detrend_plv
            )
            results[cond]["y"].append(y)
            results[cond]["phi"].append(phi)
            results[cond]["plv"].append(plv)
            results[cond]["corr"].append(corr)

    for cond in conditions:
        for k in ("y", "phi", "plv", "corr"):
            results[cond][k] = np.asarray(results[cond][k], dtype=np.float64)

    def tail_stats(arr: np.ndarray) -> Tuple[float, float]:
        tail = arr[:, -tail_len:]
        return float(np.nanmean(tail)), float(np.nanstd(tail))

    summary: Dict[str, Dict[str, float]] = {}
    for cond in conditions:
        plv_m, plv_s = tail_stats(results[cond]["plv"])
        corr_m, corr_s = tail_stats(results[cond]["corr"])
        summary[cond] = {
            "plv_tail_mean": plv_m,
            "plv_tail_std": plv_s,
            "corr_tail_mean": corr_m,
            "corr_tail_std": corr_s,
        }

    return results, summary
