from run_pipeline import run_pipeline,DATA_DIR,CH_NAMES_22,split_epochs,load_subject_epochs,reject_artifacts,FS
from fit_csp import fit_csp
import sys
import mne
import matplotlib.pyplot as plt
import numpy as np

if sys.argv[1] == 'run_pipeline':
    all_results = {}
    print("Running pipeline on all subjects...\n")

    for i in range(1, 10):
        subj = f'A0{i}'
        try:
            result = run_pipeline(subj, DATA_DIR)
            all_results[subj] = result
            print(f"  {subj}: train={result['X_train'].shape[0]:>4d} windows, "
                f"test={result['X_test'].shape[0]:>4d} windows, "
                f"rejected={result['n_rejected']:>3d} trials")
        except Exception as e:
            print(f"  {subj}: FAILED — {e}")

    print(f"\nSuccessfully processed {len(all_results)}/9 subjects")

if sys.argv[1]=='fit_csp':
    test_data, test_labels = load_subject_epochs('A01', DATA_DIR, session='T')
    _clean_data, _clean_labels, _n_rej = reject_artifacts(test_data, test_labels, threshold_uv=60.0)
    _X_tr, _X_te, _y_tr, _y_te = split_epochs(_clean_data, _clean_labels)
    _csp_tr, _csp_te, _csp_obj = fit_csp(_X_tr, _y_tr, _X_te, n_components=4)

    assert _csp_tr.ndim == 2, f"Expected 2D, got {_csp_tr.ndim}D"
    assert _csp_tr.shape[0] == _X_tr.shape[0], "Number of training samples changed"
    assert _csp_te.shape[0] == _X_te.shape[0], "Number of test samples changed"
    assert _csp_tr.shape[1] == _csp_te.shape[1], "Feature dims don't match between train and test"

    # CSP features should differ between classes
    _class_means = np.array([_csp_tr[_y_tr == c].mean(axis=0) for c in range(4)])
    assert not np.allclose(_class_means[0], _class_means[1], atol=0.1), \
        "CSP features are identical across classes — something is wrong"

    print(f"✓ fit_csp: train shape {_csp_tr.shape}, test shape {_csp_te.shape}")
    print(f"  Feature dimensionality: {_csp_tr.shape[1]}")
    print(f"  Class 0 mean features (first 4): {_class_means[0][:4].round(3)}")
    print(f"  Class 1 mean features (first 4): {_class_means[1][:4].round(3)}")

    # Visualization: CSP spatial patterns (provided)
    # Create an MNE Info object for topographic plotting
    info = mne.create_info(ch_names=CH_NAMES_22, sfreq=FS, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Plot first 4 CSP patterns
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    patterns = _csp_obj.patterns_[:4]  # First 4 patterns

    for i, (pattern, ax) in enumerate(zip(patterns, axes)):
        mne.viz.plot_topomap(pattern, info, axes=ax, show=False)
        ax.set_title(f'CSP Pattern {i+1}')

    fig.suptitle('First 4 CSP Spatial Patterns (Left Hand vs Rest)', fontsize=14)
    plt.tight_layout()
    plt.show()