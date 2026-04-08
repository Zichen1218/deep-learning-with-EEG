# Project 03 — EEG Dataset Exploration

> **Dataset:** BCI Competition IV Dataset 2a — Subject A01, training session
> **Tooling:** MNE-Python, NumPy, SciPy, Matplotlib

---

## Overview

This project loads real EEG data for the first time and walks through every step from raw GDF file to ERD maps — reusing the DSP functions built in Project 02 and applying them to actual neural signals. The goal is to understand the data structure, verify that the signal properties match expectations, and produce the core visualization that a future classifier will learn to detect.

---

## Dataset

**BCI Competition IV Dataset 2a** is a 4-class motor imagery dataset recorded from 9 subjects, each performing 288 trials across two sessions (training + evaluation). Each trial is a cue-based imagination of one of four movements:

| Class | Label | Cue |
|-------|-------|-----|
| 1 | `left_hand` | Left hand movement |
| 2 | `right_hand` | Right hand movement |
| 3 | `feet` | Both feet movement |
| 4 | `tongue` | Tongue movement |

**Recording specs:**
- 22 EEG channels + 3 EOG channels
- Sampling rate: 250 Hz
- Trial structure: 0.5s pre-cue baseline → cue onset at t=0 → 4s imagery window
- Format: GDF (data) + `.mat` (labels)

Channel layout follows the standard 10-20 system with a motor cortex focus — the central strip (C3, Cz, C4) and surrounding FC/CP rows are the most diagnostically relevant.

---

## Pipeline

Seven steps, each runnable independently:

### `load_data`
Reads `A01T.gdf` via MNE and prints basic recording metadata: sampling rate, channel count, total duration, and channel names. Entry point for sanity-checking the file loaded correctly.

### `assign_montage`
The GDF file ships with generic channel names (`EEG-0`, `EEG-C3`, `EEG-Fz`, …). This step builds a manual rename map to standard 10-20 names, sets EOG channel types, attaches the `standard_1020` montage (so MNE knows electrode positions in 3D space), bandpass-filters to **0.5–45 Hz**, and applies **average reference**. Output is `raw_eeg`: 22 EEG channels, clean and referenced.

Sanity checks: correct channel count (22), C3/C4/Cz present, montage positions non-zero.

### `plot_traces`
Plots 9 motor channels (C3, Cz, C4 + surrounding FC/CP ring) over a 10–15s window. Amplitude is in **μV** (raw MNE data is in Volts; multiply by 1e6). Y-axis clamped to ±100 μV. This is the first look at what real EEG noise looks like — irregular, low-amplitude, nothing like a textbook sine wave.

Sanity check: window shape `(9, 1250)`, max amplitude in 10–500 μV range.

### `eeg_psd`
Computes Welch PSD across all 22 channels (0.5–45 Hz, 512-point FFT). Two-panel figure:
- **Left:** all channels overlaid on a log scale, with C3 (red) and C4 (teal) highlighted
- **Right:** mean ± 1 std across channels, with EEG band shading (δ/θ/α/β/γ)

The characteristic **1/f shape** — power falling steeply with frequency — should be clearly visible. This is a universal property of neural signals and a baseline expectation before any preprocessing artifact indicates something went wrong.

Sanity check: PSD shape `(22, n_freqs)`, values in μV²/Hz range (not V²/Hz).

### `epoching`
Extracts events from annotations and segments the continuous recording into trials: `tmin=-0.5s` to `tmax=4.0s` relative to each cue onset, with baseline correction over the pre-cue window. Each epoch contains 22 channels × ~1126 timepoints.

Output shape: `(n_trials, 22, n_times)`. Sanity check verifies trial count per class and correct dimensionality.

### `topomapping`
For each of the 4 classes, averages all trials and all timepoints in the 0–4s imagery window down to a single amplitude value per channel, then renders a **scalp topography** using MNE's `plot_topomap`. The 2D head map shows which scalp regions are most active during each imagined movement.

The contralateral motor principle should appear here: left hand imagery activates the right hemisphere (around C4), right hand activates the left hemisphere (around C3). The sanity check prints C3 vs. C4 mean amplitude for both hand classes to verify this directly.

### `eeg_erd`
The main result: a **2×2 grid of ERD maps** using the spectrogram + baseline-normalization pipeline from Project 02, now applied to real trial-averaged EEG.

Grid layout:

|          | Left hand | Right hand |
|----------|-----------|------------|
| **C3**   | ERD map   | ERD map    |
| **C4**   | ERD map   | ERD map    |

For each cell: average all trials for that class and channel → STFT spectrogram (window=64 samples, overlap=60) → ERD (%) relative to the pre-cue baseline (t < 0). The expected pattern is **blue (suppression)** in the mu (8–13 Hz) and beta (13–30 Hz) bands after cue onset (t=0), strongest at the contralateral electrode.

The final printed value — mean ERD at C4 during left hand imagery in the mu band — is the scalar that a trained BCI classifier learns to detect.

---

## Project Structure

```
project-03-eeg-dataset-exploration/
├── src/
│   └── eeg_dataset_explore.py   # Full pipeline
├── mne_data/
│   └── 001-2014/
│       ├── A01T.gdf             # Raw EEG (training, subject 1)
│       └── A01T.mat             # Trial labels
├── run.py                       # CLI entry point
└── README.md
```

---

## Usage

```bash
python run.py load_data
python run.py assign_montage
python run.py plot_traces
python run.py eeg_psd
python run.py epoching
python run.py topomapping
python run.py eeg_erd
```

Steps must be run in order — each builds on the processed objects from the previous step.

---

## Requirements

```bash
pip install mne numpy scipy matplotlib
```

Download Dataset 2a from the [BCI Competition IV website](https://www.bbci.de/competition/iv/) and place GDF files in `mne_data/001-2014/`.
