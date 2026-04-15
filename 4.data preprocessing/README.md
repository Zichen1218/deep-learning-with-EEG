# EEG Preprocessing Pipeline — BCI Competition IV Dataset 2a

## File Structure

```
run_pipeline.py   — Universal preprocessing pipeline (used by ALL notebooks)
fit_csp.py        — CSP spatial filtering (used ONLY by Notebook 4: CSP+LDA/SVM)
```

---

## `run_pipeline.py`

### Usage

```python
from run_pipeline import run_pipeline, DATA_DIR

result = run_pipeline('A01', DATA_DIR)

X_train = result['X_train']   # (n_train_windows, 22, 250)
X_test  = result['X_test']    # (n_test_windows,  22, 250)
y_train = result['y_train']   # (n_train_windows,)
y_test  = result['y_test']    # (n_test_windows,)
```

### Pipeline Steps

```
load_subject_epochs()        Raw GDF → band-pass (4–38 Hz) → average reference → epochs
        ↓
reject_artifacts()           Peak-to-peak thresholding at 100 µV (data in Volts, converted internally)
        ↓
split_epochs()               Stratified train/test split at epoch level (before windowing — no leakage)
        ↓
sliding_window()             1s windows, 50% overlap: each 2s epoch → 3 windows
        ↓
compute_normalization_stats() Per-channel mean and std from training windows only
        ↓
normalize()                  Z-score normalization applied to both train and test
```

### Output shapes

| Key | Shape | Description |
|---|---|---|
| `X_train` | `(n_train_windows, 22, 250)` | Normalized training windows |
| `X_test` | `(n_test_windows, 22, 250)` | Normalized test windows |
| `y_train` | `(n_train_windows,)` | Labels: 0=left hand, 1=right hand, 2=feet, 3=tongue |
| `y_test` | `(n_test_windows,)` | Same label encoding |
| `n_rejected` | `int` | Number of artifact-rejected trials |
| `mu` | `(22,)` | Per-channel means (from training data, in Volts) |
| `sigma` | `(22,)` | Per-channel stds (from training data, in Volts) |

### Key design decisions

**Why split before windowing?**
Each 2s epoch produces 3 overlapping windows. If you split after windowing, windows from the same epoch can end up in both train and test — the model sees test data during training. This inflates accuracy by ~8 percentage points on Dataset 2a (confirmed by ablation).

**Why 100 µV artifact threshold?**
EEG data is stored in Volts by MNE. The threshold is converted internally (`* 1e6`) before comparison. Dataset 2a was pre-screened by competition organizers, so near-zero rejection rates are expected and normal.

**Why normalize with training stats only?**
In a deployed BCI, you compute statistics during calibration (training session), then apply them to all future incoming data. Using test statistics would be information leakage.

---

## `fit_csp.py`

### Usage

```python
from fit_csp import fit_csp

# Call AFTER run_pipeline — takes raw windowed data, not normalized
X_train_csp, X_test_csp, csp = fit_csp(
    result['X_train'], result['y_train'], result['X_test'],
    n_components=4
)
# X_train_csp shape: (n_train_windows, 16)  — 4 components × 4 classes (OVR)
# X_test_csp  shape: (n_test_windows,  16)
```

### Output shapes

| Variable | Shape | Description |
|---|---|---|
| `X_train_csp` | `(n_train_windows, n_components × n_classes)` | Log-variance CSP features |
| `X_test_csp` | `(n_test_windows, n_components × n_classes)` | Same feature space |
| `csp` | `mne.decoding.CSP` | Fitted CSP object (use `.patterns_` for topographic plots) |


## Dataset constants (imported from `run_pipeline.py`)

```python
from run_pipeline import DATA_DIR, FS, N_CHANNELS, N_CLASSES, CH_NAMES_22

FS          = 250    # Sampling rate (Hz)
N_CHANNELS  = 22     # EEG channels after dropping 3 EOG channels
N_CLASSES   = 4      # left hand, right hand, feet, tongue
DATA_DIR    = '../mne_data/bci_iv_2a/'
```
