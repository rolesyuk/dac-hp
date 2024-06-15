#!/usr/bin/env python3

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

f_geq = np.array([
    20, 21, 22, 23, 24, 26, 27, 29, 30, 32, 34, 36, 38, 40, 43, 45, 48, 50,
    53, 56, 59, 63, 66, 70, 74, 78, 83, 87, 92, 97, 103, 109, 115, 121, 128,
    136, 143, 151, 160, 169, 178, 188, 199, 210, 222, 235, 248, 262, 277, 292,
    309, 326, 345, 364, 385, 406, 429, 453, 479, 506, 534, 565, 596, 630, 665,
    703, 743, 784, 829, 875, 924, 977, 1032, 1090, 1151, 1216, 1284, 1357, 1433,
    1514, 1599, 1689, 1784, 1885, 1991, 2103, 2221, 2347, 2479, 2618, 2766, 2921,
    3086, 3260, 3443, 3637, 3842, 4058, 4287, 4528, 4783, 5052, 5337, 5637, 5955,
    6290, 6644, 7018, 7414, 7831, 8272, 8738, 9230, 9749, 10298, 10878, 11490,
    12137, 12821, 13543, 14305, 15110, 15961, 16860, 17809, 18812, 19871])

def load_target(filename):
    target = np.loadtxt(filename, delimiter=',', skiprows=1)
    return target

def compute_target_diff(target_filename):
    target = load_target(target_filename)
    flat_target = target.copy()
    flat_target[target[:, 0] < 1000, 1] = 0
    target_diff = target.copy()
    target_diff[:, 1] = target[:, 1] - flat_target[:, 1]
    f = interp1d(target_diff[:, 0], target_diff[:, 1], kind='quadratic')
    target_diff_geq = f(f_geq)
    return target_diff_geq

    plt.close('all')
    _, ax = plt.subplots()
    ax.plot(target[:, 0], target[:, 1], '.')
    # ax.plot(flat_target[:, 0], flat_target[:, 1], '--')
    # ax.plot(target_diff[:, 0], target_diff[:, 1], '--')
    ax.plot(f_geq, target_diff_geq, '--')
    ax.set(xscale='log')
    ax.grid()
    plt.show()

def load_geq(filename):
    with open(filename) as f:
        geq_string = f.readline().split(':')[1].strip().strip('\n')

    geq = [x.split() for x in geq_string.split('; ')]
    geq = np.array([float(x) for y in geq for x in y]).reshape((-1, 2))[:, 1]
    return geq

def save_geq(filename, geq):
    geq_string = ['GraphicEQ:' ]
    for f, a in zip(f_geq, geq):
        geq_string.append(' {:.0f} {:.1f};'.format(f, a))

    geq_string = ''.join(geq_string).strip(';')

    with open(filename, 'w') as f:
        f.write(geq_string)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_geq', dest='input_geq_filename', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'input'))
    parser.add_argument('--output_geq', dest='output_geq_filename', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--target', dest='target_filename', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'target', 'harman_in-ear_2019v2.csv'))
    parser.add_argument('--flat', dest='flat_20Hz_1kHz', type=bool, default=False)
    parser.add_argument('--lpf', dest='lpf', action='store_true', default=False)
    parser.add_argument('--lpf_filename', dest='lpf_filename', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'target', 'LPF_13000_21.txt'))
    parser.add_argument('--cutoff', dest='cutoff_frequency', type=float)
    args = parser.parse_args()

    diff = compute_target_diff(args.target_filename)
    input_geq = load_geq(args.input_geq_filename)

    if args.flat_20Hz_1kHz:
        output_geq = input_geq - diff
    else:
        output_geq = input_geq.copy()

    if args.cutoff_frequency:
        output_geq[f_geq < args.cutoff_frequency] = output_geq[f_geq >= args.cutoff_frequency][0]

    if args.lpf and args.lpf_filename:
        lpf = load_geq(args.lpf_filename)
        output_geq = output_geq + lpf

    save_geq(args.output_geq_filename, output_geq)

    plt.close('all')
    _, ax = plt.subplots(figsize=(12, 8))
    ax.plot(f_geq, input_geq, '.')
    ax.plot(f_geq, diff, '-')
    ax.plot(f_geq, output_geq, '--')
    ax.plot(np.repeat(15e3, 2), np.array([-20, 10]), '--')
    ax.set(xscale='log', xlim=[np.min(f_geq), np.max(f_geq)], ylim=[-20, 10])
    ax.grid()
    plt.show()

    exit(0)