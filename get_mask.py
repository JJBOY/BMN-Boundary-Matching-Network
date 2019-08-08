import numpy as np
import math

mask = np.zeros([100, 32, 100, 100])  # TNDT
for s in range(100):
    for d in range(100):
        e = s + d
        if e > 99:
            break
        samples = np.linspace(s - d/4, e + d/4, 32)
        for i, point in enumerate(samples):
            dec, floor = math.modf(point)
            floor = int(floor)
            if floor >= 0 and floor < 100:
                mask[floor, i, d, s] += 1 - dec
            if floor + 1 < 100 and floor + 1 >= 0:
                mask[floor + 1, i, d, s] += dec

np.save('BM_mask.npy', mask)
