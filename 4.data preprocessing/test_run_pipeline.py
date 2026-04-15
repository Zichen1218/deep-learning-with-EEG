from run_pipeline import run_pipeline,DATA_DIR

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