from ifscube import Cube

if __name__ == '__main__':
    a = Cube('ngc3081_cube.fits')
    c_opts = {'degree': 1, 'lower_threshold': 3, 'upper_threshold': 3}
    a.channel_maps(6564.0, -500.0, +500.0, log_flux=True, continuum_options=c_opts)
