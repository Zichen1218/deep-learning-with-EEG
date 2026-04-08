# Project 02 — Signal Intuition

> **EEG Deep Learning Series | Warm-up #2**
> Build the signal processing intuition you need before touching real EEG data — from a pure sine wave to an ERD map.

---

## Why This Exists

Neural signals are not images. Before applying any deep learning model to EEG, you need to think fluently in the frequency domain — to understand what a "10 Hz alpha wave" actually looks like, why FFT recovers it, what a band-pass filter does to noise, and what ERD means when a subject imagines moving their hand.

This project builds that intuition from scratch using synthetic signals only. Every concept is implemented by hand, verified with numeric sanity checks, and visualized immediately. No real data yet — just the DSP foundation that every later project assumes.

---

## What's Covered

Eight progressive checks, each building on the last:

| Check | Function | What you build |
|-------|----------|----------------|
| 1 | `show_sine` | Sine waves at alpha (10 Hz), beta (20 Hz), theta (6 Hz) |
| 2 | `show_superpose` | Linear superposition — components vs. composite |
| 3 | `noise_and_snr` | Gaussian noise at three SNR levels; SNR formula in dB |
| 4 | `show_fft` | One-sided FFT; peak recovery from a noisy composite |
| 5 | `show_bandpass_filter` | Zero-phase Butterworth band-pass filter per EEG band |
| 6 | `psd_welch` | Welch PSD; band power via trapezoidal integration |
| 7 | `show_spectrogram` | STFT spectrogram; alpha suppression during motor imagery |
| 8 | `show_erd` | ERD (%) map relative to baseline — the core BCI visualization |

---

## EEG Band Reference

These five bands are used throughout the series:

| Band  | Range (Hz) | Cognitive state |
|-------|-----------|-----------------|
| Delta | 0.5 – 4   | Deep sleep |
| Theta | 4 – 8     | Drowsiness, memory encoding |
| Alpha | 8 – 13    | Relaxed, eyes closed — **suppressed during motor imagery** |
| Beta  | 13 – 30   | Active thinking — key for motor imagery BCI |
| Gamma | 30 – 45   | High-level cognition |

Sampling rate fixed at **256 Hz** throughout (matches BCI Competition IV Dataset 2a, used in Project 03).

---

## Key Concepts Implemented

**`make_sine` / `superpose`**
Pure sine generation and linear superposition. The composite of alpha + beta + theta looks unreadable in the time domain — the FFT resolves it instantly. This gap is the core motivation for frequency-domain analysis.

**`add_noise` / `compute_snr_db`**
Zero-mean Gaussian noise added at three levels (`σ = 0.5, 1.5, 3.0`). SNR computed as `10 · log₁₀(signal_power / noise_power)`. Real EEG typically operates below 0 dB SNR — this gives you a feel for why preprocessing matters.

**`compute_fft` / `find_peaks_fft`**
One-sided magnitude spectrum via `scipy.fft`. The sanity check verifies that the three injected frequencies (6, 10, 20 Hz) and their amplitudes (0.7, 1.0, 0.5) are recovered exactly from the clean composite.

**`bandpass_filter`**
4th-order zero-phase Butterworth filter via `scipy.signal.butter` + `filtfilt`. `filtfilt` applies the filter twice (forward and backward) to eliminate phase distortion — critical for EEG where timing matters.

**`compute_psd_welch` / `band_power`**
Welch's method averages periodograms over overlapping windows to reduce variance. `band_power` integrates PSD with `np.trapz` over a frequency range — the standard way to extract scalar features per EEG band. The sanity check verifies alpha power > beta power given the signal construction.

**`compute_spectrogram`**
Short-Time Fourier Transform via `scipy.signal.spectrogram`. Window length 64 samples (250 ms at 256 Hz), 87.5% overlap. The check builds a 6-second non-stationary trial with a 2-second motor imagery segment in the middle — the spectrogram should show the alpha band going dark during that window.

**`simulate_eeg_trial` / `compute_erd`**
ERD (Event-Related Desynchronization) is the percentage power drop relative to a pre-event baseline:

```
ERD(%) = 100 × (power - baseline) / baseline
```

A 4-second trial is constructed: 0.5s baseline (mu amp=1.5, beta amp=0.8) followed by 3.5s imagery (mu amp=0.3, beta amp=0.2). The ERD map should show negative values (blue) in the mu band (8–13 Hz) after imagery onset. This is exactly what a BCI classifier detects.

---

## Project Structure

```
project-02-signal-intuition/
├── src/
│   └── signal_intuition.py   # All signal functions + visualization
├── run.py                    # CLI entry point
└── README.md
```

---

## Usage

Run any check individually from the command line:

```bash
python run.py show_sine
python run.py show_superpose
python run.py noise_and_snr
python run.py show_fft
python run.py show_bandpass_filter
python run.py psd_welch
python run.py show_spectrogram
python run.py show_erd
```

Running without arguments lists all available functions:

```bash
python run.py
```

---

## Requirements

```bash
pip install numpy scipy matplotlib
```

