import sys
from src.signal_intuition import show_sine,show_superpose,noise_and_snr,show_bandpass_filter,show_fft,psd_welch,show_spectrogram,show_erd

FUNCTIONS = {
    "show_sine":show_sine,
    "show_superpose":show_superpose,
    "noise_and_snr":noise_and_snr,
    "show_fft":show_fft,
    "show_bandpass_filter":show_bandpass_filter,
    'psd_welch':psd_welch,
    'show_spectrogram':show_spectrogram,
    'show_erd':show_erd
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
