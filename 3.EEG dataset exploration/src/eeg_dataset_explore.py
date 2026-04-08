import os
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal as sp_signal

mne.set_log_level('WARNING')

DATA_DIR   = 'mne_data/001-2014'
SUBJECT    = 'A01'
TRAIN_GDF  = os.path.join(DATA_DIR, f'{SUBJECT}T.gdf')
LABEL_MAT  = os.path.join(DATA_DIR, f'{SUBJECT}T.mat')

CLASS_NAMES  = {1:'left_hand', 2:'right_hand', 3:'feet', 4:'tongue'}
CLASS_COLORS = {1:'#2A9D8F', 2:'#E63946', 3:'#F4A261', 4:'#9F9FED'}
raw = mne.io.read_raw_gdf(os.path.join(DATA_DIR,'A01T.gdf'),stim_channel='auto',preload=True)
def load_data():
    print('=== Raw object summary ===')
    print(f'Sampling rate : {raw.info["sfreq"]} Hz')
    print(f'Channels      : {len(raw.ch_names)}')
    print(f'Duration      : {raw.times[-1]:.1f} s ({raw.times[-1]/60:.1f} min)')
    print(f'Total samples : {raw.n_times}')
    print(f'Channel names : {raw.ch_names}')
# Standard 10-20 names for Dataset 2a (in order)
EEG_CH_NAMES = [
    'Fz',
    'FC3','FC1','FCz','FC2','FC4',
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP3','CP1','CPz','CP2','CP4',
    'P1','Pz','P2','POz'
]

# ── YOUR CODE HERE ──
# Step 1: Find EEG channels in raw.ch_names (those containing 'EEG')
#print(raw.ch_names)
# Step 2: Build rename dict mapping old names → EEG_CH_NAMES
rename_map = {
    'EEG-Fz'      : 'Fz',
    'EEG-0'       : 'FC3',
    'EEG-1'       : 'FC1',
    'EEG-2'       : 'FCz',
    'EEG-3'       : 'FC2',
    'EEG-4'       : 'FC4',
    'EEG-5'       : 'C5',
    'EEG-C3'      : 'C3',
    'EEG-6'       : 'C1',
    'EEG-Cz'      : 'Cz',
    'EEG-7'       : 'C2',
    'EEG-C4'      : 'C4',
    'EEG-8'       : 'C6',
    'EEG-9'       : 'CP3',
    'EEG-10'      : 'CP1',
    'EEG-11'      : 'CPz',
    'EEG-12'      : 'CP2',
    'EEG-13'      : 'CP4',
    'EEG-14'      : 'P1',
    'EEG-Pz'      : 'Pz',
    'EEG-15'      : 'P2',
    'EEG-16'      : 'POz',
    'EOG-left'    : 'EOG1',
    'EOG-central' : 'EOG2',
    'EOG-right'   : 'EOG3',
}
# Step 3: raw.rename_channels(rename_dict)
raw.rename_channels(rename_map)
# Step 4: Set EOG channel types
raw.set_channel_types({'EOG1':'eog','EOG2':'eog','EOG3':'eog'})
# Step 5: Set montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage,match_case=False,on_missing='ignore')
# Step 6: raw_eeg = raw.copy().pick_types(eeg=True)
raw_eeg = raw.copy().pick_types(eeg=True)
# Step 7: filter 0.5–45 Hz
raw_eeg.filter(l_freq = 0.5,h_freq = 45)
# Step 8: average reference

raw_eeg.set_eeg_reference('average',projection = False)
def assign_montage():
    assert raw_eeg is not None, 'Build raw_eeg'
    assert len(raw_eeg.ch_names) == 22, \
        f'Expected 22 EEG channels after picking, got {len(raw_eeg.ch_names)}'
    assert 'C3' in raw_eeg.ch_names, 'C3 should be in ch_names after renaming'
    assert 'C4' in raw_eeg.ch_names, 'C4 should be in ch_names after renaming'
    assert 'Cz' in raw_eeg.ch_names, 'Cz should be in ch_names after renaming'

    # Montage check — positions should not be None
    pos = raw_eeg.info['chs'][0]['loc'][:3]
    assert not np.allclose(pos, 0), 'Montage not set — electrode positions are all zero'

    print('Channels after preprocessing:', raw_eeg.ch_names)
    print(f'Sampling rate: {raw_eeg.info["sfreq"]} Hz')

MOTOR_CHANNELS = ['C3','Cz','C4','FC3','FCz','FC4','CP3','CPz','CP4']

# ── YOUR CODE HERE ──
# 1. Extract data and times for MOTOR_CHANNELS
data,times = raw_eeg[MOTOR_CHANNELS,:]
# 2. Select time window 10–15s (convert to sample indices)
fs = raw_eeg.info['sfreq']
data_window = data[:,int(10*fs):int(15*fs)]
times_window = times[int(10*fs):int(15*fs)]
# 3. Plot each channel in its own subplot row, 9 rows total
# 4. Convert to μV (* 1e6), label y-axis with channel name, set ylim(-100, 100)
# 5. Set x-axis label 'Time (s)' on last subplot
def plot_traces():
    fig, axes = plt.subplots(9, 1, figsize=(12, 14), sharex=True)

    for i, (ax, ch_name) in enumerate(zip(axes, MOTOR_CHANNELS)):
        ax.plot(times_window, data_window[i] * 1e6, color='steelblue', linewidth=0.8)
        ax.set_ylabel(ch_name, fontsize=9)
        ax.set_ylim(-100, 100)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Motor channels — 10 to 15 s', fontsize=12)
    plt.tight_layout()
    plt.show()
    # ── SANITY CHECK 11 ────────────────────────────────────────────────────────────
    # Extract programmatically to verify your values
    data_all, times_all = raw_eeg[MOTOR_CHANNELS, :]
    fs_real   = int(raw_eeg.info['sfreq'])
    i0, i1    = int(10 * fs_real), int(15 * fs_real)
    window_uv = data_all[:, i0:i1] * 1e6

    assert window_uv.shape == (9, i1-i0), \
        f'Expected shape (9, {i1-i0}), got {window_uv.shape}'
    # EEG amplitude should be in the range of microvolts, not volts
    assert np.abs(window_uv).max() < 500, \
        'Values look like Volts not μV — did you multiply by 1e6?'
    assert np.abs(window_uv).max() > 0.1, \
        'Values too small — check your data extraction'
    print(f'Window shape  : {window_uv.shape}')
    print(f'Max amplitude : {np.abs(window_uv).max():.1f} μV  (expect 10–200 μV)')

# ── YOUR CODE HERE ──
# 1. psd_obj = raw_eeg.compute_psd(method='welch', fmin=0.5, fmax=45.0, n_fft=512)
# 2. freqs, psd_data (in μV²/Hz)
# 3. Two-panel figure:
#    Left:  all channels overlaid (semilogy), C3=red, C4=teal highlighted
#    Right: mean ± std (semilogy), fill_between for std band
# 4. Add EEG band shading and labels to both axes
psd_obj = raw_eeg.compute_psd(method = 'welch',fmin=0.5,fmax=45.0,n_fft=512)
freqs    = psd_obj.freqs
psd_data = psd_obj.get_data() * 1e12
BANDS = {
    'δ': (0.5,  4,  '#a8dadc'),
    'θ': (4,    8,  '#457b9d'),
    'α': (8,   13,  '#2a9d8f'),
    'β': (13,  30,  '#e9c46a'),
    'γ': (30,  45,  '#e76f51'),
}

c3_idx = raw_eeg.ch_names.index('C3')
c4_idx = raw_eeg.ch_names.index('C4')

mean_psd = psd_data.mean(axis=0)
std_psd  = psd_data.std(axis=0)
def eeg_psd():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # ── Left: all channels overlaid ──────────────────────────────────────────────
    for i, ch in enumerate(raw_eeg.ch_names):
        if ch == 'C3':
            ax1.semilogy(freqs, psd_data[i], color='red',    lw=2,   zorder=3, label='C3')
        elif ch == 'C4':
            ax1.semilogy(freqs, psd_data[i], color='teal',   lw=2,   zorder=3, label='C4')
        else:
            ax1.semilogy(freqs, psd_data[i], color='silver', lw=0.7, zorder=2, alpha=0.7)

    ax1.set_title('All channels (C3/C4 highlighted)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (μV²/Hz)')
    ax1.legend(fontsize=9)

    # ── Right: mean ± std ────────────────────────────────────────────────────────
    ax2.semilogy(freqs, mean_psd, color='steelblue', lw=2, label='mean')
    ax2.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd,
                    alpha=0.3, color='steelblue', label='±1 std')
    ax2.set_title('Mean ± std across channels')
    ax2.set_xlabel('Frequency (Hz)')

    # ── Band shading on both axes ─────────────────────────────────────────────────
    for ax in (ax1, ax2):
        for band, (flo, fhi, color) in BANDS.items():
            ax.axvspan(flo, fhi, alpha=0.15, color=color, zorder=1)
            ax.text((flo + fhi) / 2, ax.get_ylim()[0] * 1.5, band,
                    ha='center', fontsize=9, color='dimgray')

    plt.tight_layout()
    plt.show()
    # ── SANITY CHECK 12 ────────────────────────────────────────────────────────────
    assert freqs    is not None, 'Extract freqs from psd_obj'
    assert psd_data is not None, 'Extract psd_data'
    assert psd_data.shape[0] == 22, f'Expected 22 channels, got {psd_data.shape[0]}'
    assert freqs[0]  >= 0.5,  'fmin should be 0.5 Hz'
    assert freqs[-1] <= 45.0, 'fmax should be 45 Hz'

    # PSD values should be in μV²/Hz range (roughly 0.001 to 100)
    if psd_data.mean() < 1e-6:
        print('WARNING: PSD values look like V²/Hz — did you multiply by 1e12?')
    else:
        print(f'PSD shape : {psd_data.shape}  (channels × frequencies)')
        print(f'Freq range: {freqs[0]:.1f} – {freqs[-1]:.1f} Hz')
        print(f'PSD range : {psd_data.min():.4f} – {psd_data.max():.2f} μV²/Hz')

    # ── YOUR CODE HERE ──
# Step 1: events, event_id_dict = mne.events_from_annotations(raw_eeg)
# Step 2: print event_id_dict to see what codes are present
# Step 3: build event_id dict
#   Note: MNE may name them '769','770','771','772' OR 'class1','class2',...
#   Look at what event_id_dict contains and map accordingly
# Step 4: create epochs

events, event_id_dict = mne.events_from_annotations(raw_eeg)
# print(events)
# print(event_id_dict)
event_id={'left_hand':1, 'right_hand':2, 'feet':3, 'tongue':4}
epochs      = mne.Epochs(
    raw_eeg,
    events=events,
    event_id=event_id,
    tmin = -0.5,
    tmax=4.0,
    baseline=(-0.5,0.0),
    preload=True
)
def epoching():
    # ── SANITY CHECK 13 ────────────────────────────────────────────────────────────
    assert epochs   is not None, 'Create epochs'
    assert event_id is not None, 'Build event_id dict'

    X = epochs.get_data()   # (n_trials, n_channels, n_times)
    assert X.ndim == 3,     'Epoch data should be 3D: (trials, channels, times)'
    assert X.shape[1] == 22, f'Expected 22 channels, got {X.shape[1]}'

    n_times_expected = int((4.0 - (-0.5)) * raw_eeg.info['sfreq']) + 1
    assert abs(X.shape[2] - n_times_expected) <= 2, \
        f'Expected ~{n_times_expected} timepoints, got {X.shape[2]}'

    print(f'Epoch data shape : {X.shape}')
    print(f'  → {X.shape[0]} trials × {X.shape[1]} channels × {X.shape[2]} timepoints')
    print(f'Epoch time range : {epochs.tmin:.1f}s to {epochs.tmax:.1f}s')
    print('Class distribution:')
    for name in event_id:
        print(f'  {name:12s}: {len(epochs[name])} trials')

def topomapping():
    # ── YOUR CODE HERE ──
    # 4 subplots side by side, one per class
    # For each class:
    #   data = epochs[class_name].get_data()   → (n_trials, 22, n_times)
    #   mask = epochs.times >= 0
    #   mean_amp = data[:, :, mask].mean(axis=(0, 2))   → (22,)
    #   mne.viz.plot_topomap(mean_amp, epochs.info, axes=ax, show=False, cmap='RdBu_r')
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    time_mask = (epochs.times >= 0) & (epochs.times <= 4.0)

    for ax, (class_id, class_name) in zip(axes, CLASS_NAMES.items()):
        # 1. Get epoch data: (n_trials, n_channels, n_times)
        data = epochs[class_name].get_data()

        # 2. Select imagery window (0s to 4s)
        data_window = data[:, :, time_mask]

        # 3. Average over trials AND time → (n_channels,)
        mean_topo = data_window.mean(axis=0).mean(axis=1)

        # 4. Plot topomap
        mne.viz.plot_topomap(
            mean_topo,
            epochs.info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            vlim=(-2e-6, 2e-6),
            sensors=True,
            contours=4
        )
        ax.set_title(class_name, fontsize=11, color=CLASS_COLORS[class_id])

    fig.suptitle('Mean EEG topography during motor imagery (0-4s)', fontsize=13)
    plt.tight_layout()
    plt.show()
    # ── SANITY CHECK 14 ────────────────────────────────────────────────────────────
    # Verify that left and right hand have different dominant hemispheres
    lh_data = epochs['left_hand'].get_data()
    rh_data = epochs['right_hand'].get_data()
    mask    = epochs.times >= 0

    c3_idx  = epochs.ch_names.index('C3')
    c4_idx  = epochs.ch_names.index('C4')

    lh_c3   = lh_data[:, c3_idx, :][:, mask].mean()
    lh_c4   = lh_data[:, c4_idx, :][:, mask].mean()
    rh_c3   = rh_data[:, c3_idx, :][:, mask].mean()
    rh_c4   = rh_data[:, c4_idx, :][:, mask].mean()

    print(f'Left hand  — C3: {lh_c3*1e6:.3f} μV  |  C4: {lh_c4*1e6:.3f} μV')
    print(f'Right hand — C3: {rh_c3*1e6:.3f} μV  |  C4: {rh_c4*1e6:.3f} μV')
    print()
    print('Expected: left hand stronger at C4 (right hemisphere = contralateral)')
    print('Expected: right hand stronger at C3 (left hemisphere = contralateral)')
    print('If this is reversed, try more subjects — contralateral ERD is clearest in good subjects')

fs_real = int(epochs.info['sfreq'])   # 250 Hz for Dataset 2a

# ── YOUR CODE HERE ──
# Build a 2×2 grid of ERD maps:
#   rows   : C3 (top), C4 (bottom)
#   columns: left hand, right hand
#
# For each combination:
#   1. Get trial data for this class and channel
#   2. Average across trials
#   3. compute_spectrogram(..., fs=fs_real, nperseg=64, noverlap=60)
#   4. compute_erd(Sxx, t_spec, baseline_tmax=0.0)
#        baseline_tmax=0.0 means t < 0 is baseline (pre-cue)
#   5. pcolormesh on the ERD map
#      - limit frequency axis to ≤ 40 Hz
#      - use cmap='RdBu_r', vmin=-80, vmax=80
#      - draw a vertical line at t=0 (cue onset)
#      - shade the mu band (8–13 Hz) and beta band (13–30 Hz)

#====== adding functions from signal intuition =========
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
#=======================================================


def eeg_erd():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)

    classes = ['left_hand', 'right_hand']
    channels = ['C3', 'C4']

    for row, ch_name in enumerate(channels):
        ch_idx = epochs.ch_names.index(ch_name)

        for col, class_name in enumerate(classes):
            ax = axes[row, col]

            # 1. Get trial data for this class and channel
            trial_data = epochs[class_name].get_data()[:, ch_idx, :]   # (n_trials, n_times)

            # 2. Average across trials
            avg_signal = trial_data.mean(axis=0)   # (n_times,)

            # 3. Spectrogram
            freqs, t_spec, Sxx = compute_spectrogram(
                avg_signal,
                fs=fs_real,
                nperseg=64,
                noverlap=60
            )

            # 4. ERD
            erd = compute_erd(Sxx, t_spec, baseline_tmax=0.0)

            # 5. Limit frequency axis to <= 40 Hz
            freq_mask = freqs <= 40
            freqs_plot = freqs[freq_mask]
            erd_plot = erd[freq_mask, :]

            # Plot ERD map
            pcm = ax.pcolormesh(
                t_spec,
                freqs_plot,
                erd_plot,
                shading='auto',
                cmap='RdBu_r',
                vmin=-80,
                vmax=80
            )

            # Cue onset
            ax.axvline(0, color='k', linestyle='--', linewidth=1)

            # Shade mu and beta bands
            ax.axhspan(8, 13, color='gray', alpha=0.15)
            ax.axhspan(13, 30, color='yellow', alpha=0.10)

            ax.set_title(f'{class_name} — {ch_name}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_ylim(0, 40)

    fig.suptitle('ERD maps on real EEG data', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    cbar = fig.colorbar(pcm, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label('ERD (%)')

    plt.show()
    c4_idx   = epochs.ch_names.index('C4')
    lh_avg   = epochs['left_hand'].get_data()[:, c4_idx, :].mean(axis=0)
    f_r, t_r, Sxx_r = compute_spectrogram(lh_avg, fs_real, nperseg=64, noverlap=60)
    erd_r    = compute_erd(Sxx_r, t_r, baseline_tmax=0.0)

    mu_mask   = (f_r >= 8)  & (f_r <= 13)
    t_imagery = t_r > 0.5
    mean_erd  = erd_r[mu_mask][:, t_imagery].mean()

    print(f'Mean ERD at C4 during left hand imagery (mu band): {mean_erd:.1f}%')
    print()
    if mean_erd < -5:
        print('Strong ERD detected — C4 suppressed during left hand imagery ✓')
        print('This is the contralateral hemispheric response.')
        print('Your future EEGNet classifier will learn to detect exactly this pattern.')
    else:
        print('ERD is weak or absent. This can happen with some subjects or short averages.')
        print('Try subject A03 or A07 — some subjects have much cleaner ERD.')