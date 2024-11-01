import numpy as np
from numba import jit, prange, float64


@jit(float64[:](float64[:], float64[:]), nopython=True, parallel=True, cache=True)
def gauss(x, p):
    y = np.zeros_like(x)
    n = int(p.size / 3)
    for j in prange(n):
        i = j * 3
        amplitude, mean, sigma = p[i:i + 3]
        w = (x - mean) / sigma
        g = amplitude * np.exp(-w * w / 2.0)
        y += g

    return y


@jit(float64[:](float64[:], float64[:]), nopython=True, parallel=True, cache=True)
def gauss_hermite(x, p):
    y = np.zeros_like(x)
    n = int(p.size / 5)

    sq2 = np.sqrt(2)
    sq6 = np.sqrt(6)
    sq24 = np.sqrt(24)

    sq2pi = np.sqrt(2.0 * np.pi)

    for j in prange(n):
        i = j * 5
        a, l0, s, h3, h4 = p[i:i + 5]

        w = (x - l0) / s
        alpha_g = 1.0 / sq2pi * np.exp(-w ** 2 / 2.)
        hh3 = 1.0 / sq6 * (2.0 * sq2 * w ** 3 - 3.0 * sq2 * w)
        hh4 = 1.0 / sq24 * (4.0 * w ** 4 - 12.0 * w ** 2 + 3.0)

        gh = a * alpha_g / s * (1.0 + h3 * hh3 + h4 * hh4)

        y += gh

    return y


@jit(float64[:](float64[:], float64[:], float64[:]), nopython=True, parallel=True, cache=True)
def gauss_vel(x, rest_wl, p):
    c = 299792.458  # km / s
    y = np.zeros_like(x)
    n = rest_wl.size
    for j in prange(n):
        i = int(j * 3)
        amplitude, v_center, sigma = p[i:i + 3]
        lambda_ratio_squared = (x / rest_wl[j]) ** 2
        vel = c * (lambda_ratio_squared - 1.0) / (lambda_ratio_squared + 1.0)

        w = (vel - v_center) / sigma
        f_vel = amplitude * np.exp(- w * w / 2.0) / (1.0 + (vel / c))

        y += f_vel

    return y


@jit(float64[:](float64[:], float64[:], float64[:]), nopython=True, parallel=True, cache=True)
def gauss_hermite_vel(x, rest_wl, p):
    c = 299792.458  # km / s
    sq2 = np.sqrt(2)
    sq6 = np.sqrt(6)
    sq24 = np.sqrt(24)

    y = np.zeros_like(x)
    n = rest_wl.size
    for j in prange(n):
        i = int(j * 5)
        lambda_ratio_squared = (x / rest_wl[j]) ** 2
        vel = c * (lambda_ratio_squared - 1.0) / (lambda_ratio_squared + 1.0)

        a, v0, s, h3, h4 = p[i:i + 5]
        w = (vel - v0) / s

        alpha_g = np.exp(- (w * w) / 2.0)
        hh3 = ((2.0 * sq2 * (w ** 3)) - (3.0 * sq2 * w)) / sq6
        hh4 = ((4.0 * (w ** 4)) - (12.0 * (w ** 2) + 3.0)) / sq24

        # The observed flux density equals the emitted flux density divided by (1 + z)
        f_vel = a * alpha_g * (1.0 + (h3 * hh3) + (h4 * hh4)) / (1.0 + (vel / c))

        y += f_vel

    return y
