import pickle

from astropy.units import Quantity

from ifscube import Cube
from ifscube import channel_maps


def main():
    evaluate_maps = False
    if evaluate_maps:
        fw = Quantity(value=[4900, 5100], unit="Angstrom")
        cube = Cube("/home/daniel/repos/channel_outflows/test_galaxy.fits")
        cm = channel_maps.ChannelMaps(cube, fitting_window=fw)

        reference_wavelength = Quantity(value=5006.843, unit="Angstrom")
        vel_min = Quantity(value=-1000, unit="km / s")
        vel_max = Quantity(value=+1000, unit="km / s")
        channels = 10

        cm.evaluate_channel_maps(lambda0=reference_wavelength, vel_min=vel_min, vel_max=vel_max, channels=channels)
        with open("cm.pickle", "wb") as f:
            pickle.dump(cm, f)
    else:
        with open("cm.pickle", "rb") as f:
            cm = pickle.load(f)

    cm.plot_spectrum(25, 15)


if __name__ == '__main__':
    main()
