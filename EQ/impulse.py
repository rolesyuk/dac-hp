#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.interpolate import interp1d

def test_impulse(filenames):
    plt.close('all')
    _, ax = plt.subplots(2, 2, figsize=(12, 8))
    for filename in filenames:
        data, fs = sf.read(filename, dtype='float32')  
        print(filename)
        print(np.nonzero(np.diff(data, axis=-1)))

        ir = data[:, 0]
        IR = np.fft.fft(ir)

        ax[0][0].plot(np.arange(len(ir)) / fs, 20 * np.log10(np.abs(ir) + np.finfo(float).eps), '-')
        ax[0][0].set(ylim=[-92, 0])
        f = np.arange(len(IR) // 2) * fs / (len(IR) - 1)
        ax[0][1].plot(f, 20 * np.log10(np.abs(IR[:len(IR) // 2])), '.-')
        ax[0][1].set(xscale='log')
        ax[0][1].legend(filenames)
        b = np.unwrap(np.angle(IR[:len(IR) // 2]))
        a = np.vstack([f, np.ones_like(b)]).T
        x = np.linalg.lstsq(a, b, rcond=None)[0]
        ax[1][1].plot(f, b - a @ x, '.-')
        ax[1][1].set(xscale='log')
        ax[1][1].legend(filenames)

    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][1].grid()
    plt.show()

    # filename = 'lpf_13kHz_21N_48kHz.wav'
    # filename = 'ouput_impulse.wav'
    # sf.write(filename, data, fs)

    return

def generate_filter(N=21, fL=13000, fS=48000): # N - Filter length, must be odd.
    # Graphical EQ frequencies
    f_graph = np.array([
        20, 21, 22, 23, 24, 26, 27, 29, 30, 32, 34, 36, 38, 40, 43, 45, 48, 50,
        53, 56, 59, 63, 66, 70, 74, 78, 83, 87, 92, 97, 103, 109, 115, 121, 128,
        136, 143, 151, 160, 169, 178, 188, 199, 210, 222, 235, 248, 262, 277, 292,
        309, 326, 345, 364, 385, 406, 429, 453, 479, 506, 534, 565, 596, 630, 665,
        703, 743, 784, 829, 875, 924, 977, 1032, 1090, 1151, 1216, 1284, 1357, 1433,
        1514, 1599, 1689, 1784, 1885, 1991, 2103, 2221, 2347, 2479, 2618, 2766, 2921,
        3086, 3260, 3443, 3637, 3842, 4058, 4287, 4528, 4783, 5052, 5337, 5637, 5955,
        6290, 6644, 7018, 7414, 7831, 8272, 8738, 9230, 9749, 10298, 10878, 11490,
        12137, 12821, 13543, 14305, 15110, 15961, 16860, 17809, 18812, 19871])

    # Compute sinc filter.
    h = np.sinc(2 * fL / fS * (np.arange(N) - (N - 1) / 2))

    # Apply window.
    h *= np.blackman(N)

    # Normalize to get unity gain.
    h /= np.sum(h)

    f_h = np.arange((N - 1) / 2) * fS / N
    w_h = 20 * np.log10(np.abs(np.fft.fft(h))[:(N - 1) // 2])
    interp_func = interp1d(f_h, w_h, kind='quadratic')
    w_graph = interp_func(f_graph)

    plt.close('all')
    _, ax = plt.subplots(1)
    ax.plot(f_h, w_h, '*')
    ax.plot(f_graph, w_graph, '.-')
    ax.grid()
    ax.set(xlabel='Frequency (Hz)', ylabel='Attenuation (dB)')
    plt.savefig('LPF_FR_{:d}_{:d}.png'.format(fL, N), dpi=300)

    eq_csv = ['GraphicEQ: ']
    start = True
    for f in f_graph:
        if not start:
            eq_csv.append('; ')
        else:
            start = False
        eq_csv.append('{:d} '.format(f))
        eq_csv.append('{:.1f}'.format(interp_func(f)))

    eq_csv = ''.join(eq_csv)

    with open('LPF_{:d}_{:d}.txt'.format(fL, N), 'w') as f:
        f.write(eq_csv)

    # img = qrcode.make(eq_csv, error_correction=qrcode.ERROR_CORRECT_L)
    # img.save('LPF_{:d}_{:d}.png'.format(fL, N))

if __name__ == '__main__':
    filenames = ['Moondrop_Blessing_2_linphase_48kHz.wav',
                 'Moondrop_Blessing_2_linphase_48kHz_wobass.wav',
                 'Moondrop_Blessing_2_minphase_48kHz_my.wav',
                 'Moondrop_Blessing_2_minphase_48kHz.wav']
    test_impulse(filenames)

    Ns = np.arange(2, 3) * 10 + 1
    print(Ns)
    fLs = [13000]
    for fL in fLs:
        for N in Ns:
            generate_filter(N, fL)

exit(0)