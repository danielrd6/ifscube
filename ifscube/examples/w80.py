import matplotlib.pyplot as plt
from ifscube.io import line_fit
import numpy as np


def main():
    fit = line_fit.load_fit('ngc3081_cube_linefit.fits')
    w = fit.velocity_width(feature='ha', width=80, fractional_pixels=True)

    v_10 = np.zeros_like(w, dtype=float)
    v_90 = np.zeros_like(w, dtype=float)
    it = np.nditer(w, flags=['refs_ok', 'multi_index'])
    none_array = np.array([None])
    for i in it:
        if i == none_array:
            v_10[it.multi_index] = np.nan
            v_90[it.multi_index] = np.nan
        else:
            v_10[it.multi_index] = w[it.multi_index]['model_lower_velocity']
            v_90[it.multi_index] = w[it.multi_index]['model_upper_velocity']

    fig, ax = plt.subplots(ncols=2, nrows=2)
    im = ax[0, 0].imshow(fit.velocity_width_model, origin='lower')
    plt.colorbar(im, ax=ax[0, 0], orientation='vertical')
    im = ax[0, 1].imshow(v_10, origin='lower')

    ax[0, 0].set_title(r'$W_{80}$')
    ax[0, 1].set_title(r'$v_{10}$')
    plt.colorbar(im, ax=ax[0, 1], orientation='vertical')

    ax[1, 0].scatter(v_10, fit.velocity_width_model.ravel(), alpha=0.5)
    ax[1, 0].set_xlabel('$v_{10}$')
    ax[1, 0].set_ylabel('$W_{80}$')

    ax[1, 1].scatter(v_90, fit.velocity_width_model.ravel(), alpha=0.5)
    ax[1, 1].set_xlabel('$v_{90}$')

    plt.show()


if __name__ == '__main__':
    main()
