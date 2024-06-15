import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import qrcode

def generate_filter(N=59, fL=13000, fS=48000): # N - Filter length, must be odd.
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
    Ns = np.arange(2, 7) * 10 + 1
    print(Ns)
    fLs = [10000, 13000, 16000]
    for fL in fLs:
        for N in Ns:
            generate_filter(N, fL)

exit(0)