import numpy as np


def DFT_loop(signal):
    n = len(signal)
    f_signal_arr = np.zeros((n,), np.complex128)
    for u in range(n):
        fourier_signal = 0
        for x in range(n):
            euler_exp = np.exp(-2 * np.pi * u * x * 1j * (1 / n))
            fourier_signal += signal[x] * euler_exp
        f_signal_arr[u] = fourier_signal
    return f_signal_arr.astype(np.complex128)


def DFT(signal):
    n = len(signal)
    omega = np.exp(-2 * np.pi * 1j * (1 / n))
    arr = np.arange(n)
    dft_mat = np.vander(omega ** arr, increasing=True)
    return np.dot(dft_mat, signal)


def IDFT(fourier_signal):
    n = len(fourier_signal)
    omega = np.exp(2 * np.pi * 1j * (1 / n))
    arr = np.arange(n)
    idft_mat = np.vander(omega ** arr,increasing=True)
    return (1/n) * np.dot(idft_mat,fourier_signal)



