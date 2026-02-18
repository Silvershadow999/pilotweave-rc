# PilotWeave-RC

**PilotWeave-RC** is a **NumPy-only, SciPy-free** toy simulation of a *physical-reservoir-computing fabric* driven by:
- a **quasicrystalline / aperiodic sparse coupling** matrix `W`,
- sparse **event pulses** (cosmic-ray surrogate),
- a **fuzzy dark matter surrogate wave** `φ(t)` (FDM-like multi-sine),
- and coherence metrics (**rolling PLV** + rolling correlation),
with an optional **density→time-warp** and **density→curvature** modulation layer for “Janus-style” guidance experiments.

> This repository is a **simulation / conceptual exploration** (toy-model).  
> It does **not** claim experimental validation or physical propulsion.

---

## What’s inside

- `gewebe_reservoir.py`  
  Core reservoir simulation (quasicrystalline `W`, memristive readout `W_mem`, DM surrogate `φ(t)`, rolling PLV/corr, multi-seed + ablation).

- `plots.py`  
  Matplotlib visualization helpers (multi-seed stats + ablation overview).

- `main.py`  
  CLI entrypoint (baseline multi-seed + ablation study).

- `requirements.txt`  
  Minimal dependencies (`numpy`, `matplotlib`).

- `LICENSE`, `.gitignore`

Recommended optional additions:
- `CITATION.cff` (citation metadata)

---

## Core model (high-level)

- Reservoir state **x(t)** evolves under:
  - sparse quasicrystalline coupling `W @ x(t)`,
  - external event pulses `u(t)` (cosmic-ray surrogate),
  - DM surrogate coupling `α·φ(t)`,
  - and a small feedback term from memristive memory.

- Memristive weights **W_mem(t)** update via **pulse-boosted Hebbian plasticity**:
  - stronger updates when pulses occur.

- Time-consistent readout:
  - **y(t) = W_mem(t) · x(t)**

- Coherence metrics:
  - rolling Pearson correlation `corr(y, φ)`
  - rolling **PLV** (Phase Locking Value), computed via a **SciPy-free Hilbert transform** (FFT analytic signal)

- Guidance-like 2D trajectory (toy):
  - thrust direction follows DM phase
  - thrust magnitude is **gated by PLV** (only above a threshold)

---

## Quick Start

```bash
git clone https://github.com/Silvershadow999/pilotweave-rc.git
cd pilotweave-rc

pip install -r requirements.txt

# Baseline multi-seed (default: 10 seeds, 3300 steps)
python main.py multi-seed --n-seeds 10 --n-steps 3300

# Ablation study (default: 20 seeds per condition, 3300 steps)
python main.py ablation --n-seeds 20 --n-steps 3300- Configurable via dataclass (`ReservoirConfig`)  
- Multi-seed simulations with mean ± std statistics  
- Ablation studies (dm_coupling=0, hebb_pulse_boost=0, shuffle φ, φ-only)  
- PLV-gated 2D trajectories (thrust magnitude scales with instantaneous coherence)  
- CLI interface (`main.py`) with subcommands: `multi-seed` & `ablation`

## How it works (one paragraph)

1. Reservoir state `x(t)` evolves under quasicrystalline coupling `W` + event-driven input pulses.  
2. Memristive-like weight vector `W_mem(t)` is updated via pulse-boosted Hebbian plasticity.  
3. Readout is computed time-consistently: `y(t) = W_mem(t) · x(t)`.  
4. Phase-locking between `y(t)` and the DM-surrogate wave `φ(t)` is quantified via rolling PLV.  
5. A 2D "guidance-like" trajectory applies thrust proportional to `y(t)` only when PLV exceeds a threshold (`min_plv_thr`).

## Quick Start

```bash
git clone https://github.com/Silvershadow999/pilotweave-rc.git
cd pilotweave-rc

# Install dependencies
pip install -r requirements.txt

# Run baseline multi-seed (10 seeds, 3300 steps)
python main.py multi-seed --n-seeds 10 --n-steps 3300

# Run ablation study (20 seeds, 3300 steps)
python main.py ablation --n-seeds 20 --n-steps 3300
