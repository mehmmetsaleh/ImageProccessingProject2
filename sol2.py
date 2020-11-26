import numpy as np
import scipy.io.wavfile as sci
from skimage import color
from skimage import io

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import fftpack, ndimage


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
        f_im[:, col_idx] = DFT(image[:, col_idx])
    for row_idx in range(image.shape[0]):
        f_im[row_idx, :] = DFT(f_im[row_idx, :])
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


def change_samples(filename, ratio):
    rate, data = sci.read(filename)
    new_data = resize(data, ratio)
    sci.write("change_samples.wav", rate, new_data)


def resize(data, ratio):
    ori_samples = DFT(data)
    shifted_ori_samples = np.fft.fftshift(ori_samples)
    n_samples = len(shifted_ori_samples)
    if ratio > 1:
        n_new_samples = int(n_samples / ratio)
        print(n_new_samples)
        diff = n_samples - n_new_samples
        resized_arr = np.zeros(ori_samples.shape)
        if diff % 2 == 0:
            resized_arr[diff // 2: len(resized_arr) - diff // 2 + 1] = shifted_ori_samples[
                diff // 2: n_samples - diff // 2 + 1]
        else:
            resized_arr[diff // 2: len(resized_arr) - diff // 2] = shifted_ori_samples[
                diff // 2: n_samples - diff // 2]
        return IDFT(np.fft.ifftshift(resized_arr)).astype(data.dtype)

    elif ratio < 1:
        



if __name__ == '__main__':
    change_samples("aria_4kHz.wav",2)


    # img = color.rgb2gray(io.imread('monkey.jpg'))
    # imgplot = plt.imshow(img,cmap="gray")
    # plt.show()
    # img2 = DFT2(img)
    # fft2 = fftpack.fft2(img)
    # imgplot = plt.imshow(np.log10(abs(img2)))
    # plt.magnitude_spectrum(img2.flatten())
    # plt.show()
