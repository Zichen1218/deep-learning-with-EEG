import sys
from src.eeg_dataset_explore import load_data,assign_montage,plot_traces,eeg_psd,epoching,topomapping,eeg_erd
FUNCTIONS = {
    'load_data':load_data,
    'assign_montage':assign_montage,
    'plot_traces':plot_traces,
    'eeg_psd':eeg_psd,
    'epoching':epoching,
    'topomapping':topomapping,
    'eeg_erd':eeg_erd
}

def main():
    if len(sys.argv)==1:
        print('available functions: ')
        for func in FUNCTIONS:
            print(f" {func} ")
    else:
        func = sys.argv[1]
        try:
            FUNCTIONS[func]()
        except KeyError:
            print('wrong function name.')

if __name__ == "__main__":
    main()
