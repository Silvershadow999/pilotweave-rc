# PilotWeave-RC  
**Pilot-wave surfing on fuzzy dark matter surrogate waves**  
*Ein SciPy-freies, reines NumPy-basiertes Physical Reservoir Computing Toy-Modell*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-powered-orange?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy--free-success?style=for-the-badge" alt="SciPy-free">
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge" alt="MIT">
</p>

<p align="center">
  Sparse kosmische Pulse + Fuzzy-Dark-Matter-Surrogat-Welle → PLV-gesteuerte Trajektorien
</p>

---

## Was macht PilotWeave-RC besonders?

- **Quasikristallines Reservoir** (Golden-Ratio-sparse Kopplung)
- **Memristives Kurzzeitgedächtnis** (puls-verstärkte Hebbian-Plastizität)
- **Fuzzy Dark Matter (FDM) Surrogat** als globale Phase-Referenz
- **PLV-Gating**: Schub / Thrust nur bei hoher Phasenkohärenz
- **2D-Trajektorien** wie ein Surfer, der der Pilotwelle folgt
- Komplett **SciPy-frei** → nur NumPy + matplotlib
- Eigener schneller rolling PLV via FFT-Hilbert
- Multi-seed + Ablationsstudien (baseline, shuffle, dm0, boost0, phi_only)
- Optionale **Dichte-modulierte Substeps** (Zeitkompression bei hoher Aktivität)

**Wichtig:** Das ist ein **reines Simulations- und Konzept-Experiment** – kein echter Antrieb, kein validierter physikalischer Effekt.

## Quick Start – sofort loslegen (Copy & Paste)

```bash
# 1. Repository klonen
git clone https://github.com/Silvershadow999/pilotweave-rc.git
cd pilotweave-rc

# 2. Minimale Abhängigkeiten installieren
pip install numpy matplotlib

# ────────────────────────────────────────────────
# Schnell-Experimente – wähle eine Zeile und los!

# A) Standard Multi-Seed + schöne Trajektorie-Übersicht
python main.py multi-seed --n-seeds 8 --save-plot multi-8seeds.png

# B) Ablationsstudie (zeigt welche Komponenten wirklich wichtig sind)
python main.py ablation --n-seeds 12 --save-plot ablation-12seeds.png

# C) Mit Dichte- & Krümmungs-Modulation (experimentell, Zeitkompression)
python main.py multi-seed --enable-density --log-density --n-seeds 6 --save-plot density-6seeds.png

# D) Kleiner Testlauf (schnell, für Debugging / Laptop)
python main.py multi-seed --n-seeds 3 --n-steps 1200 --window 300 --save-plot quick-test.png
