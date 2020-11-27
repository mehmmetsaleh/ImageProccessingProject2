import numpy as np
import scipy.io.wavfile as sci
import skimage.color as sk
from skimage import io
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


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
    im = np.zeros(fourier_image.shape, dtype=np.complex128)
    for col_idx in range(fourier_image.shape[1]):
        im[:, col_idx] = IDFT(fourier_image[:, col_idx])
    for row_idx in range(fourier_image.shape[0]):
        im[row_idx, :] = IDFT(im[row_idx, :])
    return im.astype(np.complex128)


def change_rate(filename, ratio):
    rate, data = sci.read(filename)
    sci.write("change_rate.wav", int(rate * ratio), data)


def change_samples(filename, ratio):
    rate, data = sci.read(filename)
    new_data = resize(data, ratio)
    sci.write("change_samples.wav", rate, new_data)
    return new_data.astype(np.float64)


def resize(data, ratio):
    f_on_ori_samples = DFT(data)
    shifted_ori_samples = np.fft.fftshift(f_on_ori_samples)
    n_samples = len(shifted_ori_samples)
    n_new_samples = int(n_samples / ratio)
    if ratio > 1:
        diff = n_samples - n_new_samples
        if diff % 2 == 0:
            new_arr = shifted_ori_samples[diff // 2: n_samples - diff // 2]
        else:
            new_arr = shifted_ori_samples[diff // 2 + 1: n_samples - diff // 2]
        return IDFT(np.fft.ifftshift(new_arr)).astype(data.dtype)

    elif ratio < 1:
        diff = n_new_samples - n_samples
        new_arr = shifted_ori_samples
        if diff % 2 == 0:
            new_arr = np.pad(new_arr, (diff // 2, diff // 2), 'constant')
        else:
            new_arr = np.pad(new_arr, (diff // 2, diff // 2 + 1), 'constant')
        return IDFT(np.fft.ifftshift(new_arr)).astype(data.dtype)

    else:
        return data


def resize_spectrogram(data, ratio):
    spec = stft(data)
    new_array = np.zeros((spec.shape[0], int(spec.shape[1] / ratio)))
    for i in range(spec.shape[0]):
        new_array[i, :] = resize(spec[i, :], ratio)
    new_arr = istft(new_array)
    return new_arr.astype(data.dtype)


def resize_vocoder(data, ratio):
    spec = stft(data)
    corrected_spec = phase_vocoder(spec, ratio)
    new_arr = istft(corrected_spec)
    return new_arr.astype(data.dtype)


def conv_der(im):
    x_kernel = np.array([[0.5, 0, -0.5]])
    y_kernel = np.reshape(x_kernel, (3, 1))
    dx = signal.convolve2d(im, x_kernel, 'same')
    dy = signal.convolve2d(im, y_kernel, 'same')
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def fourier_der(im):
    f1 = DFT2(im)
    f1 = np.fft.fftshift(f1)

    # derivative in X direction
    u_arr_x = np.array([i - f1.shape[0] // 2 for i in range(f1.shape[0])])
    v_mask = np.empty(f1.shape)
    for col in range(v_mask.shape[1]):
        v_mask[:, col] = u_arr_x
    v_f1 = np.multiply(f1, v_mask)
    x_der = ((2 * np.pi * 1j) / v_f1.shape[0]) * IDFT2(v_f1)

    # derivative in Y direction
    u_arr_y = np.array([i - f1.shape[1] // 2 for i in range(f1.shape[1])])
    h_mask = np.empty(f1.shape)
    for row in range(h_mask.shape[0]):
        h_mask[row, :] = u_arr_y
    h_f1 = np.multiply(f1, h_mask)
    y_der = ((2 * np.pi * 1j) / h_f1.shape[1]) * IDFT2(h_f1)

    magnitude = np.sqrt(np.abs(x_der) ** 2 + np.abs(y_der) ** 2)
    return magnitude.astype(np.float64)


def read_image(filename, representation):
    im = io.imread(filename)
    im_float = im.astype(np.float64)
    im_float /= 255

    if representation == 1:
        im_g = sk.rgb2gray(im_float)
        return im_g
    elif representation == 2:
        return im_float


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
