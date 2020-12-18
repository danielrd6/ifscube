import pkg_resources

from ifscube import channel_maps
from ifscube import datacube


def test_flux_conservation():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc3081_cube.fits')
    cube = datacube.Cube(file_name)

    cm = channel_maps.ChannelMaps(cube)
    cm.set_channels(6562.8, -500, +500, n_channels=10)
    cm.evaluate_maps(fit_continuum=True, continuum_windows=[6300, 6400, 6620, 6680], flux_threshold=0)
    c1 = cm.maps.sum()

    cm.set_channels(6562.8, -500, +500, n_channels=30)
    cm.evaluate_maps(fit_continuum=True, continuum_windows=[6300, 6400, 6620, 6680], flux_threshold=0)
    c2 = cm.maps.sum()

    assert abs((c2 - c1) / c1) < 0.01
