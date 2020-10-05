import matplotlib.pyplot as plt
from ifscube.io import line_fit
import numpy as np


def main():
    fit = line_fit.load_fit('ngc3081_cube_linefit.fits')
    w = fit.velocity_width(feature='ha', width=80)

    v_10 = np.zeros_like(w, dtype=float)
    it = np.nditer(w, flags=['refs_ok', 'multi_index'])
    none_array = np.array([None])
    for i in it:
        if i == none_array:
            v_10[it.multi_index] = np.nan
        else:
            v_10[it.multi_index] = w[it.multi_index]['model_lower_velocity']

    fig, ax = plt.subplots(ncols=2, nrows=1)
    im = ax[0].imshow(fit.velocity_width_model, origin='lower')
    plt.colorbar(im, ax=ax[0], orientation='horizontal')
    im = ax[1].imshow(v_10, origin='lower')
    plt.colorbar(im, ax=ax[1], orientation='horizontal')

    ax[0].set_title(r'$W_{80}$')
    ax[1].set_title(r'$V_{10}$')
    plt.show()


if __name__ == '__main__':
    main()
