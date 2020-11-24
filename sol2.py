import numpy as np


def DFT(signal):
    n = len(signal)
    f_signal_arr = np.zeros((n,), np.complex128)
    for u in range(n):
        fourier_signal = 0
        for x in range(n):
            euler_exp = np.exp(-2 * np.pi * u * x * 1j * (1 / n))
            fourier_signal += signal[x] * euler_exp
        f_signal_arr[u] = fourier_signal
    return f_signal_arr.astype(np.complex128)


def IDFT(fourier_signal):
    pass
