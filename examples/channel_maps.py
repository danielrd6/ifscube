from src.ifscube import Cube
from src.ifscube.channel_maps import channelmaps

if __name__ == '__main__':
    a = Cube('ngc3081_cube.fits')
    c_opts = {'degree': 1, 'lower_threshold': 3, 'upper_threshold': 3}
    cm = channelmaps(a, 6564.0, -500.0, +500.0, log_flux=True, continuum_options=c_opts)
