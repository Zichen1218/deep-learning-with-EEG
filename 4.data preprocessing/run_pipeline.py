import numpy as np
import mne
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')


DATA_DIR = '../mne_data/bci_iv_2a/'

# Dataset constants
FS = 250          # Sampling rate (Hz)
N_CHANNELS = 22   # EEG channels (after dropping EOG)
N_CLASSES = 4      # left hand, right hand, feet, tongue
EVENT_IDS = {'left_hand': 769, 'right_hand': 770, 'feet': 771, 'tongue': 772}

# Channel names for the 22 EEG channels in Dataset 2a
CH_NAMES_22 = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

def load_subject_epochs(subject_id, data_dir, session='T'):
    """
    Load and epoch one subject's data from BCI Competition IV Dataset 2a.
    
    Args:
        subject_id: str, e.g. 'A01'
        data_dir:   str, path to directory containing .gdf and .mat files
        session:    str, 'T' for training, 'E' for evaluation
    
    Returns:
        epochs_data: np.ndarray, shape (n_trials, 22, n_times)
        labels:      np.ndarray, shape (n_trials,), values in {0, 1, 2, 3}
                     0=left hand, 1=right hand, 2=feet, 3=tongue
    """
    gdf_path = f"{data_dir}/{subject_id}{session}.gdf"
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    
    # Rename channels to standard 10-20 names
    eeg_ch_names = raw.ch_names[:22]
    rename_map = {old: new for old, new in zip(eeg_ch_names, CH_NAMES_22)}
    raw.rename_channels(rename_map)
    
    # Pick only EEG channels
    raw.pick_channels(CH_NAMES_22)
    
    # Set montage for topographic plotting
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # Band-pass filter: 4-38 Hz
    raw.filter(4.0, 38.0, fir_design='firwin')
    
    # Average reference
    raw.set_eeg_reference('average', projection=False)
    
    # Extract events
    events, event_id_full = mne.events_from_annotations(raw)
    
    # Map to our event IDs — GDF annotations vary by MNE version,
    # so we find the correct annotation keys for codes 769-772
    target_events = {}
    for key, val in event_id_full.items():
        # Try to extract the numeric code from the annotation string
        try:
            code = int(key)
        except ValueError:
            # Some MNE versions use strings like '769', others differ
            continue
        if code in [769, 770, 771, 772]:
            target_events[key] = val
    
    # Create epochs: 0.5s to 2.5s relative to cue onset
    epochs = mne.Epochs(
        raw, events, event_id=target_events,
        tmin=0.5, tmax=2.5,
        baseline=None,  # We'll do baseline correction manually
        preload=True,
        proj=False
    )
    
    epochs_data = epochs.get_data()  # (n_trials, 22, n_times)
    
    # Get labels
    if session == 'T':
        # Training file: labels from events
        event_codes = epochs.events[:, 2]
        # Map MNE event IDs back to original codes
        inv_map = {v: int(k) for k, v in target_events.items()}
        original_codes = np.array([inv_map[c] for c in event_codes])
        labels = original_codes - 769  # 769->0, 770->1, 771->2, 772->3
    else:
        # Evaluation file: labels from .mat file
        mat_path = f"{data_dir}/{subject_id}E.mat"
        mat = loadmat(mat_path)
        labels = mat['classlabel'].flatten() - 1  # 1-indexed to 0-indexed
        # Trim to match number of valid epochs
        labels = labels[:len(epochs_data)]
    
    # Baseline correction: subtract mean of first 125 samples (0.5s)
    baseline = epochs_data[:, :, :125].mean(axis=2, keepdims=True)
    epochs_data = epochs_data - baseline
    
    print(f"  {subject_id}{session}: {epochs_data.shape[0]} epochs, "
          f"shape {epochs_data.shape}, "
          f"class dist: {np.bincount(labels)}")
    
    return epochs_data, labels

def detect_artifacts(epochs_data, threshold_uv=100.0):
    """
    Detect artifact-contaminated trials using peak-to-peak thresholding.
    
    Args:
        epochs_data:  np.ndarray, shape (n_trials, n_channels, n_times)
        threshold_uv: float, PTP threshold in microvolts
    
    Returns:
        clean_mask: np.ndarray, shape (n_trials,), dtype=bool
                    True = clean, False = artifact
    """
    # YOUR CODE HERE
    # mask = []
    # n_trials, n_channels, n_times = epochs_data.shape
    # for i in range(n_trials):
    #     clean = True
    #     for j in range(n_channels):
    #         if np.max(epochs_data[i,j,:]) - np.min(epochs_data[i,j,:]) >threshold_uv:
    #             clean = False
    #             break
    #     mask.append(clean)
    # return np.array(mask)
    ptp_per_channel = np.max(epochs_data, axis=2) - np.min(epochs_data, axis=2)
    max_ptp_per_trial = np.max(ptp_per_channel, axis=1) *1e6
    clean_mask = max_ptp_per_trial <= threshold_uv
    return clean_mask

def reject_artifacts(epochs_data, labels, threshold_uv=100.0):
    """
    Reject artifact trials and their corresponding labels.
    
    Args:
        epochs_data:  np.ndarray, shape (n_trials, n_channels, n_times)
        labels:       np.ndarray, shape (n_trials,)
        threshold_uv: float, PTP threshold in microvolts
    
    Returns:
        clean_data:   np.ndarray, shape (n_clean, n_channels, n_times)
        clean_labels: np.ndarray, shape (n_clean,)
        n_rejected:   int
    """
    # YOUR CODE HERE
    mask = detect_artifacts(epochs_data,threshold_uv)
    clean_data = epochs_data[mask]
    clean_labels = labels[mask]
    n_rejected = (~mask).sum()
    return clean_data,clean_labels,n_rejected

def split_epochs(epochs_data, labels, test_size=0.2, random_state=42):
    """
    Stratified train/test split at the epoch level.
    
    Args:
        epochs_data:  np.ndarray, shape (n_trials, n_channels, n_times)
        labels:       np.ndarray, shape (n_trials,)
        test_size:    float
        random_state: int
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # YOUR CODE HERE
    sss = StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state)
    train_idx,test_idx = next(sss.split(epochs_data,labels))
    X_train,X_test = epochs_data[train_idx],epochs_data[test_idx]
    y_train,y_test = labels[train_idx],labels[test_idx]
    return X_train, X_test, y_train, y_test


def sliding_window(X, y, window_size=250, step_size=125):
    """
    Extract overlapping sliding windows from epochs.
    
    Args:
        X:           np.ndarray, shape (n_epochs, n_channels, n_times)
        y:           np.ndarray, shape (n_epochs,)
        window_size: int, samples per window
        step_size:   int, step between windows
    
    Returns:
        X_windows: np.ndarray, shape (n_windows, n_channels, window_size)
        y_windows: np.ndarray, shape (n_windows,)
    """
    # YOUR CODE HERE
    n_epochs,n_channels,n_times = X.shape
    windows = []
    labels = []

    for i in range(n_epochs):
        for start in range(0,n_times-window_size+1,step_size):
            windows.append(X[i,:,start:start+window_size])
            labels.append(y[i])
    windows,labels = map(np.array,[windows,labels])
    return windows,labels

def compute_normalization_stats(X_train):
    """
    Compute per-channel mean and std from training data.
    
    Args:
        X_train: np.ndarray, shape (n_windows, n_channels, n_times)
    
    Returns:
        mu:    np.ndarray, shape (n_channels,)
        sigma: np.ndarray, shape (n_channels,)
    """
    # YOUR CODE HERE
    mu = np.mean(X_train,axis=(0,2))
    sigma = np.std(X_train,axis =(0,2))
    return mu,sigma


def normalize(X, mu, sigma):
    """
    Apply z-score normalization using precomputed statistics.
    
    Args:
        X:     np.ndarray, shape (n_windows, n_channels, n_times)
        mu:    np.ndarray, shape (n_channels,)
        sigma: np.ndarray, shape (n_channels,)
    
    Returns:
        X_norm: np.ndarray, same shape as X
    """
    # YOUR CODE HERE
    mu = mu.reshape(1,-1,1)
    sigma = sigma.reshape(1,-1,1)
    X = (X-mu)/sigma
    return X

def run_pipeline(subject_id, data_dir, session='T', artifact_threshold=100.0,
                 test_size=0.2, window_size=250, step_size=125, random_state=42):
    """
    Complete preprocessing pipeline: load → reject → split → window → normalize.
    
    Args:
        subject_id:          str, e.g. 'A01'
        data_dir:            str, path to data directory
        session:             str, 'T' or 'E'
        artifact_threshold:  float, PTP threshold in µV
        test_size:           float, fraction for test set
        window_size:         int, samples per window
        step_size:           int, step between windows
        random_state:        int, random seed
    
    Returns:
        dict with 'X_train', 'X_test', 'y_train', 'y_test',
             'n_rejected', 'mu', 'sigma'
    """
    # YOUR CODE HERE
    data, labels = load_subject_epochs(subject_id, data_dir, session)
    clean_data,clean_labels,n_rejected = reject_artifacts(data,labels,threshold_uv=artifact_threshold)
    X_train, X_test, y_train, y_test = split_epochs(clean_data,clean_labels,test_size=test_size,random_state=random_state)
    train_windows, train_labels = sliding_window(X_train,y_train,window_size=window_size,step_size=step_size)
    test_windows,test_labels = sliding_window(X_test,y_test,window_size=window_size,step_size=step_size)
    mu,sigma = compute_normalization_stats(train_windows)
    X_fin_train = normalize(train_windows,mu,sigma)
    X_fin_test = normalize(test_windows,mu,sigma)
    return {'X_train':X_fin_train, 'X_test':X_fin_test, 'y_train':train_labels, 'y_test':test_labels,
             'n_rejected':n_rejected, 'mu':mu, 'sigma':sigma}
    

