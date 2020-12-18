from ifscube.channel_maps import ChannelMaps
from ifscube.datacube import Cube


def main():
    a = Cube('ngc3081_cube.fits')

    cm = ChannelMaps(a)
    n_channels = 12
    cm.set_channels(6562.8, -300, +300, n_channels=n_channels)

    continuum_opts = {'degree': 1, 'lower_threshold': 3, 'upper_threshold': 3}
    cm.evaluate_maps(fit_continuum=True, continuum_windows=[6300, 6400, 6620, 6670], flux_threshold=1e-17,
                     continuum_options=continuum_opts)
    maps, axes, pmaps = cm.plot()


if __name__ == '__main__':
    main()
