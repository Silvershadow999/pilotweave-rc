# PilotWeave-RC  
**Pilot-wave surfing on fuzzy dark matter (surrogate)**

A SciPy-free physical reservoir computing model that leverages sparse cosmic ray pulses and a fuzzy dark matter *surrogate* wave to study strong phase coherence (rolling Pearson correlation + PLV) and PLV-modulated guidance-like trajectories.

**Author:** Alexandra-Nicole Anna Drinda (Silvershadow999)  
**License:** MIT (see [LICENSE](LICENSE))

> **Note / Disclaimer**  
> This repository is a **computational simulation** exploring signal coherence and guidance-style control signals in a physical reservoir computing context.  
> It does **not** claim experimentally validated propulsion, verified coupling to dark matter, or any real-world physical effect beyond the simulated surrogate model.

## Core Concept

PilotWeave-RC simulates a thin, adaptive metamaterial "fabric" that:

- Uses a quasicrystalline aperiodic coupling matrix for high-capacity, noise-resistant reservoir dynamics  
- Employs memristive/ferroelectric Hebbian plasticity (pulse-boosted learning triggered by sparse cosmic ray events)  
- Maintains time-consistent readout: `y(t) = W_mem(t) · x(t)`  
- Couples to a global fuzzy-DM pilot wave *surrogate*  
- Generates thrust in a **guidance-like (Bohm-inspired)** equation **only when** phase-locking (PLV) is high

**Goal:** Investigate self-calibrating, ultra-low-energy attitude and trajectory control **signals** for satellites and deep-space vehicles (pure simulation study).

## Features

- Fully NumPy-based (no SciPy dependency)  
- Configurable via dataclass (`ReservoirConfig`)  
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
