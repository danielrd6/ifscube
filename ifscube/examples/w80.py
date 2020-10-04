import matplotlib.pyplot as plt
from ifscube.io import line_fit


def main():
    fit = line_fit.load_fit('ngc3081_cube_linefit.fits')
    fit.velocity_width(feature='ha', width=80)

    fig, ax = plt.subplots()
    im = ax.imshow(fit.velocity_width_model, origin='lower')
    plt.colorbar(im, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
