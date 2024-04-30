import numpy as np


def gauss_vel(x, rest_wl, p):
    c = 299792.458  # km / s
    y = np.zeros_like(x)
    j = 0
    for i in range(0, len(p), 3):
        try:
            lam_ratio = (x / rest_wl[j])
        except IndexError:
            lam_ratio = (x / rest_wl)
        vel = c * (lam_ratio - 1.0) / (lam_ratio + 1.0)

        w = -((vel - p[i + 1]) / p[i + 2]) ** 2 / 2.0
        f_vel = p[i] * np.exp(w) / (1.0 + (vel / c))

        y = y + f_vel
        j = j + 1

    return y
