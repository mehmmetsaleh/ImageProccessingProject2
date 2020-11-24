import numpy as np
import scipy.io.wavfile as sci


# def DFT_loop(signal):
#     n = len(signal)
#     f_signal_arr = np.zeros((n,), np.complex128)
#     for u in range(n):
#         fourier_signal = 0
#         for x in range(n):
#             euler_exp = np.exp(-2 * np.pi * u * x * 1j * (1 / n))
#             fourier_signal += signal[x] * euler_exp
#         f_signal_arr[u] = fourier_signal
#     return f_signal_arr.astype(np.complex128)


def DFT(signal):
    n = len(signal)
    omega = np.exp(-2 * np.pi * 1j * (1 / n))
    arr = np.arange(n)
    dft_mat = np.vander(omega ** arr, increasing=True)
    return np.dot(dft_mat, signal).astype(np.complex128)


def IDFT(fourier_signal):
    n = len(fourier_signal)
    omega = np.exp(2 * np.pi * 1j * (1 / n))
    arr = np.arange(n)
    idft_mat = np.vander(omega ** arr, increasing=True)
    res = (1 / n) * np.dot(idft_mat, fourier_signal)
    return res.astype(np.complex128)


def DFT2(image):
    f_im = np.zeros(image.shape, dtype=np.complex128)
    for col_idx in range(image.shape[1]):
        f_im[:, col_idx] = DFT(image[col_idx])
    for row_idx in range(image.shape[0]):
        f_im[row_idx, :] = DFT(f_im[row_idx:, ])
    return f_im.astype(np.complex128)


def IDFT2(fourier_image):
    im = np.zeros(fourier_image.shape, dtype=np.float64)
    for col_idx in range(fourier_image.shape[1]):
        im[:, col_idx] = IDFT(fourier_image[:, col_idx])
    for row_idx in range(fourier_image.shape[0]):
        im[row_idx, :] = IDFT(im[row_idx, :])
    return im.astype(np.float64)


def change_rate(filename, ratio):
    rate, data = sci.read(filename)
    sci.write("change_rate.wav", int(rate * ratio), data)


