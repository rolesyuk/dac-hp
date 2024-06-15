#/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os 

targets = []
targets.append([os.path.join('..', 'AutoEq', 'compensation'),
                ['autoeq_in-ear', 'diffuse_field', 'harman_in-ear_2019v2', 'harman_in-ear_2019v2_wo_bass']])
targets.append([os.path.join('..', 'AutoEq', 'measurements', 'crinacle', 'resources'),
                ['crinacle_harman_in-ear_2019v2_wo_bass']])

plt.close('all')
_, ax = plt.subplots(figsize=(12, 8))
legend = []

for target_group in targets:
    for target in target_group[1]:
        x = np.loadtxt(os.path.join(target_group[0], target + '.csv'), delimiter=',', skiprows=1)
        ax.plot(x[:, 0], x[:, 1])
    legend = legend + target_group[1]

ax.set(xlabel='Frequency (Hz)', xscale='log')
ax.grid()
ax.legend(legend)
plt.show()

exit(0)