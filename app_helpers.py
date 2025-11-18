from scipy.signal import find_peaks, butter, filtfilt
import numpy as np

def standardize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def remove_outliers(signal, thresh = 3):
    return signal[np.abs(signal - np.mean(signal)) < thresh * np.std(signal)]

def butterworth_filter(signal, lowcut, highcut, fs, order=4):
    '''"Biometrics Analysis V.1"
    Apply a Butterworth band-pass filter to the signal.
    Parameters:
    signal : array-like
        The input signal to be filtered.
    lowcut : float
        The low cutoff frequency of the filter, in Hz.
    highcut : float
        The high cutoff frequency of the filter, in Hz.
    fs : float
        The sampling frequency of the signal.
    order : int, optional
        The order of the filter. Default is 4.
    Returns:
    array-like
        The filtered signal.
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, signal)
    return y