import matplotlib.pyplot as plt
from ifscube.io import line_fit
import numpy as np

from astropy.units import Quantity


def main():
    fit = line_fit.load_fit('ngc3081_cube_linefit.fits')
    w = fit.velocity_width(feature='ha', width=80, fractional_pixels=True, rest_wavelength=Quantity(6562.8, 'angstrom'))

    v_10 = np.zeros_like(w, dtype=float)
    v_90 = np.zeros_like(w, dtype=float)
    centroid = np.zeros_like(w, dtype=float)
    it = np.nditer(w, flags=['refs_ok', 'multi_index'])
    none_array = np.array([None])
    for i in it:
        if i == none_array:
            v_10[it.multi_index] = np.nan
            v_90[it.multi_index] = np.nan
            centroid[it.multi_index] = np.nan
        else:
            v_10[it.multi_index] = w[it.multi_index]['model_lower_velocity']
            v_90[it.multi_index] = w[it.multi_index]['model_upper_velocity']
            centroid[it.multi_index] = w[it.multi_index]['model_centroid_velocity']

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(231)

    im = ax.imshow(fit.velocity_width_model, origin='lower')
    plt.colorbar(im, ax=ax, orientation='vertical')
    ax.set_title(r'$W_{80}$ (km/s)')

    ax = fig.add_subplot(232)
    im = ax.imshow(v_10, origin='lower')
    plt.colorbar(im, ax=ax, orientation='vertical')
    ax.set_title(r'$v_{10}$ (km/s)')

    ax = fig.add_subplot(233)
    im = ax.imshow(centroid, origin='lower', cmap='Spectral_r')
    plt.colorbar(im, ax=ax, orientation='vertical')
    ax.set_title(r'Centroid velocity (km/s)')

    ax = fig.add_subplot(234)
    ax.scatter(v_10, fit.velocity_width_model.ravel(), alpha=0.5)
    ax.set_xlabel('$v_{10}$')
    ax.set_ylabel('$W_{80}$')

    ax = fig.add_subplot(235)
    ax.scatter(v_90, fit.velocity_width_model.ravel(), alpha=0.5)
    ax.set_xlabel('$v_{90}$')

    plt.show()


if __name__ == '__main__':
    main()
