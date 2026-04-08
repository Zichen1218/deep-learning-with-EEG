import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
import scipy.io as sio
import os

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.size':        11,
})

# EEG band definitions — commit these to memory, you will use them forever
EEG_BANDS = {
    'delta': (0.5,  4),   # deep sleep
    'theta': (4,    8),   # drowsiness, memory
    'alpha': (8,   13),   # relaxed, eyes closed
    'beta':  (13,  30),   # active thinking — most important for motor imagery
    'gamma': (30,  45),   # high-level cognition
}
BAND_COLORS = {
    'delta': '#9F9FED', 'theta': '#F4A261',
    'alpha': '#2A9D8F', 'beta':  '#E63946', 'gamma': '#8338EC'
}
FS = 256        # sampling rate in Hz — fixed for the whole notebook

def make_sine(t, freq, amplitude=1.0, phase=0.0):
    """
    Generate a pure sine wave.

    Parameters
    ----------
    t         : np.ndarray, shape (N,)  — time vector in seconds
    freq      : float                   — frequency in Hz
    amplitude : float                   — peak amplitude (default 1.0)
    phase     : float                   — phase offset in radians (default 0.0)

    Returns
    -------
    np.ndarray, shape (N,)

    Pseudocode
    ----------
    return amplitude * sin(2π * freq * t + phase)
    """
    # ── YOUR CODE HERE ──
    return amplitude *np.sin(2*np.pi*t*freq+phase)


# Build the time vector: 2 seconds at 256 Hz
# np.linspace(start, stop, num, endpoint=False)
#   start    = 0
#   stop     = DURATION
#   num      = FS * DURATION   (total number of samples)
#   endpoint = False means stop is NOT included — avoids duplicate at period boundary
DURATION   = 2.0
# ── YOUR CODE HERE ──
t          = np.linspace(0,DURATION,int(DURATION*FS),endpoint=False)   # shape (512,)
alpha_wave = make_sine(t,10)   # 10 Hz, amplitude 1.0, phase 0
beta_wave  = make_sine(t,20,0.5)   # 20 Hz, amplitude 0.5, phase 0
theta_wave = make_sine(t,6,0.7,np.pi/4)   # 6 Hz,  amplitude 0.7, phase π/4
def show_sine():
    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    for ax, wave, label, color in zip(axes,
        [alpha_wave, beta_wave, theta_wave],
        ['Alpha 10 Hz (A=1.0)', 'Beta 20 Hz (A=0.5)', 'Theta 6 Hz (A=0.7, phase=π/4)'],
        [BAND_COLORS['alpha'], BAND_COLORS['beta'], BAND_COLORS['theta']]):
        ax.plot(t[:FS], wave[:FS], color=color, linewidth=1.8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-1.2, 1.2)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Check 1 — sine waves', fontweight='bold')
    plt.tight_layout()
    plt.show()

def superpose(*waves):
    """
    Sum any number of signals together.

    Parameters
    ----------
    *waves : np.ndarray  — any number of arrays, all same shape

    Returns
    -------
    np.ndarray  — element-wise sum

    Pseudocode
    ----------
    result = zeros_like(waves[0])
    for each wave:
        result += wave
    return result
    """
    # ── YOUR CODE HERE ──
    return np.sum(waves,axis=0)


# ── YOUR CODE HERE ──
composite = superpose(alpha_wave,beta_wave,theta_wave)   # sum of alpha + beta + theta

def show_superpose():
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for wave, color, label in zip(
        [alpha_wave, beta_wave, theta_wave],
        [BAND_COLORS['alpha'], BAND_COLORS['beta'], BAND_COLORS['theta']],
        ['Alpha', 'Beta', 'Theta']):
        axes[0].plot(t[:FS], wave[:FS], alpha=0.6, linewidth=1.2, color=color, label=label)
    axes[0].set_title('Individual components')
    axes[0].legend(fontsize=9)
    axes[1].plot(t[:FS], composite[:FS], color='#264653', linewidth=1.5)
    axes[1].set_title('Composite — can you identify the three waves by eye?')
    axes[1].set_xlabel('Time (s)')
    fig.suptitle('Check 2 — superposition', fontweight='bold')
    plt.tight_layout()
    plt.show()
def add_noise(sig, noise_std, seed=42):
    """
    Add zero-mean Gaussian white noise to a signal.

    Parameters
    ----------
    sig       : np.ndarray  — clean signal
    noise_std : float       — standard deviation σ of the noise
    seed      : int         — random seed for reproducibility

    Returns
    -------
    np.ndarray  — noisy signal, same shape as input

    Pseudocode
    ----------
    rng   = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=noise_std, size=sig.shape)
    return sig + noise
    """
    # ── YOUR CODE HERE ──
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0,scale=noise_std,size=sig.shape)
    return sig+noise


def compute_snr_db(clean, noisy):
    """
    Compute Signal-to-Noise Ratio in decibels.

    Parameters
    ----------
    clean : np.ndarray  — original signal without noise
    noisy : np.ndarray  — signal with noise added

    Returns
    -------
    float  — SNR in dB

    Pseudocode
    ----------
    noise         = noisy - clean
    signal_power  = mean(clean ** 2)
    noise_power   = mean(noise ** 2)
    return 10 * log10(signal_power / noise_power)
    """
    # ── YOUR CODE HERE ──
    noise = noisy-clean
    return 10 *np.log10(np.mean(clean**2)/np.mean(noise**2))


# ── YOUR CODE HERE ──
noisy_low    = add_noise(composite,0.5)   # noise_std = 0.5
noisy_medium = add_noise(composite,1.5)   # noise_std = 1.5
noisy_high   = add_noise(composite,3.0)   # noise_std = 3.0
def noise_and_snr():
    snr_low    = compute_snr_db(composite, noisy_low)
    snr_medium = compute_snr_db(composite, noisy_medium)
    snr_high   = compute_snr_db(composite, noisy_high)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, sig, label in zip(axes,
        [composite, noisy_low, noisy_medium, noisy_high],
        ['Clean', f'Low noise (SNR={snr_low:.1f} dB)',
        f'Medium noise (SNR={snr_medium:.1f} dB)',
        f'High noise (SNR={snr_high:.1f} dB)']):
        ax.plot(t[:FS], sig[:FS], color='#264653', linewidth=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-8, 8)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Check 3 — noise and SNR', fontweight='bold')
    plt.tight_layout()
    plt.show()
    print(f'SNR: low={snr_low:.1f} | medium={snr_medium:.1f} | high={snr_high:.1f} dB')
def compute_fft(sig, fs):
    """
    Compute one-sided FFT magnitude spectrum of a real signal.

    Parameters
    ----------
    sig : np.ndarray, shape (N,)  — input signal
    fs  : float                   — sampling rate in Hz

    Returns
    -------
    freqs      : np.ndarray, shape (N//2,)  — frequency axis in Hz
    magnitudes : np.ndarray, shape (N//2,)  — amplitude at each frequency

    Pseudocode
    ----------
    N          = len(sig)
    yf         = fft(sig)
    freqs_all  = fftfreq(N, 1/fs)
    half       = N // 2
    magnitudes = (2.0 / N) * abs(yf[:half])
    return freqs_all[:half], magnitudes
    """
    # ── YOUR CODE HERE ──
    N = len(sig)
    yf = fft(sig)
    freqs_all = fftfreq(N,1/fs)
    half = N//2
    magnitudes = (2.0/N)*abs(yf[:half])
    return freqs_all[:half],magnitudes

    


def find_peaks_fft(freqs, magnitudes, n_peaks=3, min_magnitude=0.1):
    """
    Find the dominant frequency peaks in an FFT magnitude spectrum.

    Parameters
    ----------
    freqs         : np.ndarray  — frequency axis
    magnitudes    : np.ndarray  — magnitudes
    n_peaks       : int         — how many peaks to return
    min_magnitude : float       — ignore peaks below this value

    Returns
    -------
    list of (freq_hz, magnitude) tuples, sorted by magnitude descending

    Pseudocode
    ----------
    peak_indices = sp_signal.find_peaks(magnitudes, height=min_magnitude)[0]
    sort peak_indices by magnitudes[peak_indices] descending
    keep top n_peaks
    return [(freqs[i], magnitudes[i]) for i in top_indices]
    """
    # ── YOUR CODE HERE ──
    peak_indices = sp_signal.find_peaks(magnitudes,height=min_magnitude)[0]
    sort_indices = peak_indices[np.argsort(magnitudes[peak_indices])[::-1]]
    top_indices = sort_indices[:n_peaks]
    return [(freqs[i], magnitudes[i]) for i in top_indices]
def show_fft():
    freqs_c, mags_c = compute_fft(composite, FS)
    assert freqs_c is not None and mags_c is not None, 'compute_fft returned None'
    assert len(freqs_c) == len(composite) // 2, 'Expected N//2 frequency bins'
    assert freqs_c[0] == 0.0, 'First bin = 0 Hz (DC component)'

    peaks = find_peaks_fft(freqs_c, mags_c, n_peaks=3)
    detected = sorted([round(p[0]) for p in peaks])
    assert detected == [6, 10, 20], f'Expected [6, 10, 20] Hz, got {detected}'
    peak_dict = {round(f): m for f, m in peaks}
    assert abs(peak_dict[10] - 1.0) < 0.05, f'Alpha A ≈ 1.0, got {peak_dict[10]:.3f}'
    assert abs(peak_dict[20] - 0.5) < 0.05, f'Beta  A ≈ 0.5, got {peak_dict[20]:.3f}'
    assert abs(peak_dict[6]  - 0.7) < 0.05, f'Theta A ≈ 0.7, got {peak_dict[6]:.3f}'

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(t[:FS], composite[:FS], color='#264653', linewidth=1.5)
    axes[0].set_title('Time domain — unreadable')
    axes[0].set_xlabel('Time (s)')
    axes[1].plot(freqs_c, mags_c, color='#264653', linewidth=1.5)
    for freq, mag in peaks:
        axes[1].annotate(f'{freq:.0f} Hz\nA={mag:.2f}',
                        xy=(freq, mag), xytext=(freq+1.5, mag+0.04),
                        fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.2))
    axes[1].set_xlim(0, 50)
    axes[1].set_title('Frequency domain — three peaks perfectly recovered')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    fig.suptitle('Check 4 — FFT', fontweight='bold')
    plt.tight_layout()
    plt.show()
    print(f'Peaks: {[(f"{f:.0f}Hz", f"{m:.2f}") for f, m in peaks]}')
def bandpass_filter(sig, lowcut, highcut, fs, order=4):
    """
    Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    sig     : np.ndarray  — input signal
    lowcut  : float       — lower cutoff in Hz
    highcut : float       — upper cutoff in Hz
    fs      : float       — sampling rate in Hz
    order   : int         — filter order (default 4)

    Returns
    -------
    np.ndarray  — filtered signal, same shape as input

    Pseudocode
    ----------
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = sp_signal.butter(order, [low, high], btype='band')
    return sp_signal.filtfilt(b, a, sig)
    """
    # ── YOUR CODE HERE ──
    nyq = fs/2
    low = lowcut/nyq
    high = highcut/nyq
    b,a = sp_signal.butter(order,[low,high],btype='band')
    return sp_signal.filtfilt(b,a,sig)


# Filter noisy_medium through each EEG band
# Skip delta — its 0.5 Hz lower bound needs a much longer signal to be reliable
# ── YOUR CODE HERE ──
# filtered_bands: dict mapping band name → filtered np.ndarray
filtered_bands = {
    'alpha':bandpass_filter(alpha_wave,8,13,FS),
    'beta':bandpass_filter(beta_wave,13,30,FS)
}
def show_bandpass_filter():
    assert len(filtered_bands) > 0, 'filtered_bands is empty'
    assert 'alpha' in filtered_bands, 'alpha band missing'
    assert 'beta'  in filtered_bands, 'beta band missing'
    f_a, m_a = compute_fft(filtered_bands['alpha'], FS)
    f_b, m_b = compute_fft(filtered_bands['beta'],  FS)
    assert 8  <= f_a[np.argmax(m_a)] <= 13, 'Alpha peak should be 8–13 Hz'
    assert 13 <= f_b[np.argmax(m_b)] <= 30, 'Beta peak should be 13–30 Hz'

    bands_plot = [b for b in ['theta','alpha','beta','gamma'] if b in filtered_bands]
    fig, axes  = plt.subplots(len(bands_plot)+1, 2, figsize=(13, 10))
    fig.suptitle('Check 5 — band-pass filtering', fontweight='bold')
    axes[0,0].plot(t[:FS], noisy_medium[:FS], color='#264653', linewidth=0.9)
    axes[0,0].set_ylabel('Noisy input', fontsize=9)
    f_n, m_n = compute_fft(noisy_medium, FS)
    axes[0,1].plot(f_n, m_n, color='#264653', linewidth=1.0)
    axes[0,1].set_xlim(0,50)
    for row, band in enumerate(bands_plot, 1):
        col = BAND_COLORS[band]
        f, m = compute_fft(filtered_bands[band], FS)
        axes[row,0].plot(t[:FS], filtered_bands[band][:FS], color=col, linewidth=1.2)
        axes[row,0].set_ylabel(band, fontsize=9)
        axes[row,1].plot(f, m, color=col, linewidth=1.2)
        axes[row,1].set_xlim(0,50)
    axes[0,0].set_title('Time domain')
    axes[0,1].set_title('Frequency domain')
    axes[-1,0].set_xlabel('Time (s)')
    axes[-1,1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
def compute_psd_welch(sig, fs, nperseg=256, noverlap=None):
    """
    Compute Power Spectral Density using Welch's method.

    Parameters
    ----------
    sig      : np.ndarray  — input signal
    fs       : float       — sampling rate in Hz
    nperseg  : int         — samples per window (default 256)
    noverlap : int|None    — overlap in samples (None → nperseg // 2)

    Returns
    -------
    freqs : np.ndarray  — frequency axis in Hz, range [0, fs/2]
    psd   : np.ndarray  — power density (amplitude² / Hz)

    Pseudocode
    ----------
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, psd
    """
    # ── YOUR CODE HERE ──
    if noverlap is None:
        noverlap = nperseg // 2
    freqs,psd = sp_signal.welch(sig,fs,nperseg = nperseg,noverlap=noverlap)
    return freqs,psd


def band_power(freqs, psd, low, high):
    """
    Integrate PSD over a frequency band to get total band power.

    Parameters
    ----------
    freqs : np.ndarray  — frequency axis
    psd   : np.ndarray  — PSD values
    low   : float       — lower bound in Hz
    high  : float       — upper bound in Hz

    Returns
    -------
    float  — total power in the band

    Pseudocode
    ----------
    mask = (freqs >= low) & (freqs <= high)
    return np.trapz(psd[mask], freqs[mask])
    """
    # ── YOUR CODE HERE ──
    mask = (freqs >= low)&(freqs<=high)
    return np.trapz(psd[mask],freqs[mask])
def psd_welch():
    # ── SANITY CHECK 6 ─────────────────────────────────────────────────────────────
    f_psd, psd_clean = compute_psd_welch(composite,    FS, nperseg=256)
    f_psd, psd_noisy = compute_psd_welch(noisy_medium, FS, nperseg=256)
    assert f_psd    is not None, 'compute_psd_welch returned None'
    assert f_psd[0] == 0.0,      'First bin should be 0 Hz'
    assert f_psd[-1] == FS/2,    f'Last bin should be Nyquist {FS/2} Hz'

    bp_alpha = band_power(f_psd, psd_clean, 8, 13)
    bp_beta  = band_power(f_psd, psd_clean, 13, 30)
    assert bp_alpha is not None, 'band_power returned None'
    assert bp_alpha > bp_beta, 'Alpha power > beta power (alpha amplitude is larger)'

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Check 6 — PSD Welch', fontweight='bold')
    for psd, label, color in [(psd_clean,'Clean','#2A9D8F'),(psd_noisy,'Noisy','#E63946')]:
        axes[0].semilogy(f_psd, psd, label=label, color=color, linewidth=2)
    for band,(lo,hi) in EEG_BANDS.items():
        axes[0].axvspan(lo, hi, alpha=0.07, color=BAND_COLORS[band])
        axes[0].text((lo+hi)/2, axes[0].get_ylim()[0]*1.5, band[0].upper(),
                    ha='center', fontsize=9, color=BAND_COLORS[band], fontweight='bold')
    axes[0].set_xlim(0,45)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power (amplitude²/Hz)')
    axes[0].set_title('PSD: notice the 1/f shape and the flat noise floor')
    axes[0].legend(fontsize=9)

    band_names  = [b for b in EEG_BANDS if b != 'delta']
    pw_clean = [band_power(f_psd, psd_clean, *EEG_BANDS[b]) for b in band_names]
    pw_noisy = [band_power(f_psd, psd_noisy, *EEG_BANDS[b]) for b in band_names]
    x = np.arange(len(band_names))
    axes[1].bar(x-0.2, pw_clean, 0.4, label='Clean', color='#2A9D8F', alpha=0.8)
    axes[1].bar(x+0.2, pw_noisy, 0.4, label='Noisy', color='#E63946', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(band_names)
    axes[1].set_ylabel('Band power')
    axes[1].set_title('White noise raises ALL bands uniformly')
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    print(f'Alpha power={bp_alpha:.4f}  Beta power={bp_beta:.4f}')
def compute_spectrogram(sig, fs, nperseg=64, noverlap=56):
    """
    Compute a spectrogram (time-frequency power map).

    Parameters
    ----------
    sig      : np.ndarray  — input signal
    fs       : float       — sampling rate in Hz
    nperseg  : int         — window length in samples (default 64)
    noverlap : int         — overlap in samples (default 56 = 87.5%)

    Returns
    -------
    f     : np.ndarray, shape (F,)    — frequency axis in Hz
    t_seg : np.ndarray, shape (T,)    — time axis in seconds
    Sxx   : np.ndarray, shape (F, T)  — power at each (freq, time)

    Pseudocode
    ----------
    f, t_seg, Sxx = sp_signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t_seg, Sxx
    """
    # ── YOUR CODE HERE ──
    f,t_seg,Sxx = sp_signal.spectrogram(sig,fs,nperseg=nperseg,noverlap=noverlap)
    return f, t_seg,Sxx


# Build the non-stationary 6-second trial
# Hint: build three 2-second segments using make_sine() and add_noise(),
#       then np.concatenate() them.
# Rest segments:  theta (6Hz, A=0.8) + alpha (10Hz, A=1.2) + noise(std=0.5)
# Imagery segment: theta (6Hz, A=0.8) + alpha (10Hz, A=0.15) + noise(std=0.5)

T_long  = 6.0
t_long  = np.linspace(0, T_long, int(FS * T_long), endpoint=False)
t_seg2  = t_long[:int(FS * 2)]   # 2-second time vector for each segment

# ── YOUR CODE HERE ──
seg1 = make_sine(t_seg2,10,1.2)+make_sine(t_seg2,6,0.8)
seg1 = add_noise(seg1,0.5)
seg2 = make_sine(t_seg2,10,0.15)+make_sine(t_seg2,6,0.8)
seg2 = add_noise(seg2,0.5)
seg3 = make_sine(t_seg2,10,1.2)+make_sine(t_seg2,6,0.8)
seg3 = add_noise(seg3,0.5)
trial_signal = np.concatenate([seg1,seg2,seg3])
def show_spectrogram():
    # ── SANITY CHECK 7 ─────────────────────────────────────────────────────────────
    assert trial_signal is not None, 'Build trial_signal'
    assert len(trial_signal) == int(FS * T_long), \
        f'Expected {int(FS*T_long)} samples, got {len(trial_signal)}'

    f_sp, t_sp, Sxx = compute_spectrogram(trial_signal, FS, nperseg=64, noverlap=56)
    assert Sxx is not None,               'compute_spectrogram returned None'
    assert Sxx.shape[0] == len(f_sp),     'Rows must match frequency axis'
    assert Sxx.shape[1] == len(t_sp),     'Cols must match time axis'

    alpha_mask = (f_sp >= 8) & (f_sp <= 13)
    alpha_rest  = Sxx[alpha_mask][:, t_sp <  2].mean()
    alpha_imag  = Sxx[alpha_mask][:, (t_sp >= 2) & (t_sp < 4)].mean()
    assert alpha_imag < alpha_rest, \
        'Alpha power must be lower during imagery — check your signal construction'

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle('Check 7 — spectrogram', fontweight='bold')
    axes[0].plot(t_long, trial_signal, color='#264653', linewidth=0.8)
    axes[0].axvspan(2, 4, alpha=0.15, color='#E63946', label='Imagery (ERD)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(fontsize=9, loc='upper right')
    fm = f_sp <= 45
    im = axes[1].pcolormesh(t_sp, f_sp[fm], 10*np.log10(Sxx[fm]+1e-12),
                            shading='gouraud', cmap='RdYlBu_r')
    axes[1].axvspan(2, 4, alpha=0.15, color='#E63946')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Alpha band (8–13 Hz) should go dark/blue during imagery')
    plt.colorbar(im, ax=axes[1], label='Power (dB)')
    plt.tight_layout()
    plt.show()
    print(f'Alpha power — rest: {alpha_rest:.5f} | imagery: {alpha_imag:.5f}')
def simulate_eeg_trial(t, mu_amp, beta_amp, noise_std=0.8, seed=0):
    """
    Simulate an EEG-like trial with controllable mu and beta power.

    Parameters
    ----------
    t         : np.ndarray  — time vector
    mu_amp    : float       — amplitude of mu rhythm (simulates 10 Hz)
    beta_amp  : float       — amplitude of beta rhythm (simulates 20 Hz)
    noise_std : float       — noise standard deviation
    seed      : int         — random seed

    Returns
    -------
    np.ndarray  — simulated EEG-like signal

    Pseudocode
    ----------
    rng   = np.random.default_rng(seed)
    delta = make_sine(t, 2,  amplitude=2.0, phase=rng.uniform(0, np.pi))
    theta = make_sine(t, 6,  amplitude=1.0, phase=rng.uniform(0, np.pi))
    mu    = make_sine(t, 10, amplitude=mu_amp,   phase=rng.uniform(0, np.pi))
    beta  = make_sine(t, 20, amplitude=beta_amp, phase=rng.uniform(0, np.pi))
    noise = noise_std * rng.standard_normal(len(t))
    return delta + theta + mu + beta + noise
    """
    # ── YOUR CODE HERE ──
    rng = np.random.default_rng(seed)
    delta = make_sine(t,2,amplitude=2.0,phase=rng.uniform(0,np.pi))
    theta = make_sine(t,6,amplitude=1.0,phase=rng.uniform(0,np.pi))
    mu = make_sine(t,10,amplitude=mu_amp,phase=rng.uniform(0,np.pi))
    beta = make_sine(t,20,amplitude=beta_amp,phase = rng.uniform(0,np.pi))
    noise = noise_std*rng.standard_normal(len(t))
    return delta+theta+mu+beta+noise

def compute_erd(Sxx, t_spec, baseline_tmax=0.0):
    """
    Compute ERD (%) from a spectrogram relative to a pre-event baseline.

    Parameters
    ----------
    Sxx           : np.ndarray, shape (F, T)  — spectrogram power matrix
    t_spec        : np.ndarray, shape (T,)    — time axis in seconds
    baseline_tmax : float  — use columns where t_spec <= this as baseline

    Returns
    -------
    erd : np.ndarray, shape (F, T)  — ERD % at each (freq, time)

    Pseudocode
    ----------
    bl_mask  = t_spec <= baseline_tmax           # boolean mask for baseline cols
    baseline = Sxx[:, bl_mask].mean(axis=1, keepdims=True)  # shape (F, 1)
    erd      = 100 * (Sxx - baseline) / (baseline + 1e-30)  # avoid div by zero
    return erd
    """
    # ── YOUR CODE HERE ──
    bl_mask = t_spec <=baseline_tmax
    baseline = Sxx[:,bl_mask].mean(axis=1,keepdims=True)
    erd = 100*(Sxx-baseline)/(baseline+1e-30)
    return erd


# Build a 4-second trial:
#   0.0 – 0.5s : baseline (rest)  → mu_amp=1.5, beta_amp=0.8
#   0.5 – 4.0s : imagery          → mu_amp=0.3, beta_amp=0.2  (ERD)
# Hint: build two segments (0.5s baseline, 3.5s imagery), concatenate

# ── YOUR CODE HERE ──
t_base = np.linspace(0,0.5,int(0.5*FS),endpoint=False)
t_imag = np.linspace(0,3.5,int(3.5*FS),endpoint=False)
baseline_seg = simulate_eeg_trial(t_base,mu_amp=1.5,beta_amp=0.8)
imag_seg = simulate_eeg_trial(t_imag,mu_amp=0.3,beta_amp=0.2)
t_trial   = t_trial = np.linspace(0, 4.0, int(FS * 4.0), endpoint=False)   # 4-second time vector
eeg_trial = np.concatenate([baseline_seg,imag_seg])   # concatenated baseline + imagery
def show_erd():
    # ── SANITY CHECK 8 ─────────────────────────────────────────────────────────────
    assert eeg_trial is not None, 'Build eeg_trial'
    assert len(eeg_trial) == int(FS * 4.0), f'Expected {int(FS*4)} samples'

    f_e, t_e, Sxx_e = compute_spectrogram(eeg_trial, FS, nperseg=64, noverlap=60)
    erd_map = compute_erd(Sxx_e, t_e, baseline_tmax=0.5)
    assert erd_map is not None,                  'compute_erd returned None'
    assert erd_map.shape == Sxx_e.shape,         'ERD shape must match Sxx'

    mu_mask  = (f_e >= 8) & (f_e <= 13)
    erd_mean = erd_map[mu_mask][:, t_e > 0.5].mean()
    assert erd_mean < 0, f'Mean ERD should be negative during imagery, got {erd_mean:.1f}%'

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Check 8 — ERD map (the core BCI visualization)', fontweight='bold')
    axes[0].plot(t_trial, eeg_trial, color='#264653', linewidth=0.8)
    axes[0].axvline(0.5, color='black', linewidth=1.5, linestyle='--', label='Imagery onset')
    axes[0].axvspan(0.5, 4.0, alpha=0.08, color='#E63946')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].legend(fontsize=9)
    fm = f_e <= 45
    im = axes[1].pcolormesh(t_e, f_e[fm], erd_map[fm],
                            cmap='RdBu_r', vmin=-80, vmax=80, shading='auto')
    axes[1].axvline(0.5, color='black', linewidth=1.5, linestyle='--')
    axes[1].axhspan(8,  13, alpha=0.15, color='#2A9D8F', label='Mu (8–13 Hz)')
    axes[1].axhspan(13, 30, alpha=0.10, color='#E63946', label='Beta (13–30 Hz)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Blue = ERD (suppression) — your classifier will detect this')
    axes[1].legend(fontsize=9, loc='upper right')
    plt.colorbar(im, ax=axes[1], label='ERD (%)')
    plt.tight_layout()
    plt.show()
    print(f'Mean ERD during imagery (mu band): {erd_mean:.1f}%')