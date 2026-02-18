# PilotWeave-RC
**Pilot-wave surfing on fuzzy dark matter (surrogate simulation)**

A SciPy-free, NumPy-based physical reservoir computing toy model that combines **sparse event pulses** (“cosmic-ray surrogate”) with a **fuzzy dark matter (FDM) surrogate wave** φ(t) to study **phase coherence** (rolling Pearson correlation + rolling PLV) and **PLV-gated guidance-like trajectories**.

**Author:** Alexandra-Nicole Anna Drinda (Silvershadow999)  
**License:** MIT

> **Disclaimer (important)**  
> This repository is a **computational toy model / simulation**.  
> It does **not** claim experimentally validated propulsion, verified coupling to dark matter, or any real-world physical effect beyond the surrogate model used here.

---

## Core Idea (in 5 lines)

- A quasicrystalline sparse coupling matrix **W** drives rich reservoir dynamics.
- Sparse random event pulses act as “cosmic-ray surrogate” perturbations + background noise.
- A global surrogate field φ(t) (FDM-like) acts as a reference wave.
- A time-consistent readout is computed as: **y(t) = W_mem(t) · x(t)** with memristive-like plasticity.
- Coherence between y(t) and φ(t) is tracked via **rolling correlation** and **rolling PLV**; trajectory thrust is **gated by PLV**.

---

## Features

- **Pure NumPy** (SciPy-free): FFT-based Hilbert transform for PLV
- **Configurable** via `ReservoirConfig` dataclass
- **Multi-seed experiments** with mean ± std statistics
- **Five-condition ablation study**:
  - `baseline`
  - `shuffle_phi`
  - `dm0` (dm_coupling = 0)
  - `boost0` (hebb_pulse_boost = 0)
  - `phi_only` (no event pulses)
- **Guidance-like 2D trajectories**: thrust only when PLV exceeds a threshold
- **CLI** interface via `main.py`

---

## Repository Structure

---

## Installation

```bash
git clone https://github.com/Silvershadow999/pilotweave-rc.git
cd pilotweave-rc
pip install -r requirements.txt

python main.py multi-seed --n-seeds 10 --n-steps 3300
python main.py ablation --n-seeds 20 --n-steps 3300

python main.py multi-seed --n-seeds 10 --n-steps 3300 --save-plot out_multiseed.png

python main.py ablation   --n-seeds 20 --n-steps 3300 --save-plot out_ablation.png
