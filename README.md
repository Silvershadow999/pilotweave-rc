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
- Ablation studies (dm_coupling=0, hebb_pulse_boost=0, shuffle φ, φ-# PilotWeave-RC
**Pilot-wave surfing on fuzzy dark matter (surrogate simulation)**

A SciPy-free physical reservoir computing model that leverages sparse cosmic ray pulses and a fuzzy dark matter *surrogate* wave to study strong phase coherence (rolling Pearson correlation + PLV) and PLV-modulated guidance-like trajectories.

**Author:** Alexandra-Nicole Anna Drinda (Silvershadow999)  
**License:** MIT (see [LICENSE](LICENSE))

> **Note / Disclaimer**
> This repository is a **computational simulation / toy model** exploring signal coherence and guidance-style control signals in a physical reservoir computing context.
> It does **not** claim experimentally validated propulsion, verified coupling to dark matter, or any real-world physical effect beyond the simulated surrogate model.

---

## What’s inside

PilotWeave-RC simulates an adaptive “fabric” (reservoir) that:

- Uses a **quasicrystalline, sparse coupling matrix** for robust high-capacity dynamics
- Learns via **memristive / ferroelectric-style Hebbian plasticity** (pulse-boosted by sparse “cosmic ray” events)
- Keeps the readout **time-consistent**: `y(t) = W_mem(t) · x(t)`
- Couples to a global **fuzzy-DM surrogate wave** `φ(t)`
- Measures coherence via:
  - rolling Pearson correlation `corr(y, φ)`
  - rolling **PLV** (Phase Locking Value) using a SciPy-free FFT-Hilbert analytic signal
- Generates **guidance-like 2D trajectories** only when PLV is high (**PLV gate**)

### Optional “density/curvature” extension (simulation-only)
Some branches/variants add a “density” proxy ρ(t) from pulses + reservoir activity and use it to:
- modulate internal substeps (“effective framerate”)
- modulate a “curvature factor” that scales thrust
This is explicitly **simulation-only** and intended as an exploratory knob.

---

## Install

```bash
git clone https://github.com/Silvershadow999/pilotweave-rc.git
cd pilotweave-rc
pip install -r requirements.txt
