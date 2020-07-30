import pkg_resources
import pytest

from ifscube import onedspec, modeling, datacube


def simple_fit(function: str = 'gaussian', fit_type: str = 'spectrum', **kwargs):
    if fit_type == 'spectrum':
        file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
        spec = onedspec.Spectrum(file_name)
        fit = modeling.LineFit(spec, function=function, fitting_window=(6400.0, 6700.0), **kwargs)
    elif fit_type == 'cube':
        file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc3081_cube.fits')
        cube = datacube.Cube(file_name)
        fit = modeling.LineFit3D(cube, function=function, fitting_window=(6400.0, 6700.0), **kwargs)
    else:
        raise RuntimeError(f'fit_type "{fit_type}" not understood.')
    fit.add_feature(name='n2_6548', rest_wavelength=6548.04, amplitude='mean', velocity=0.0, sigma=100.0)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude='mean', velocity=0.0, sigma=100.0)
    fit.add_feature(name='n2_6583', rest_wavelength=6583.46, amplitude='mean', velocity=0.0, sigma=100.0)
    return fit


def full_fit(function: str = 'gaussian'):
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
    spec = onedspec.Spectrum(file_name)
    fit = modeling.LineFit(spec, function=function, fitting_window=(6400.0, 6800.0))
    names = ['n2_6548', 'ha', 'n2_6583', 's2_6716', 's2_6731']
    r_wl = [6548.04, 6562.8, 6583.46, 6716.44, 6730.86]

    for name, wl in zip(names, r_wl):
        fit.add_feature(name=name, rest_wavelength=wl, amplitude=1.0e-14, velocity=0.0, sigma=100.0,
                        kinematic_group=0)

    for name, wl in zip(names, r_wl):
        fit.add_feature(name=name + '_b', rest_wavelength=wl, amplitude=1.0e-14, velocity=-200.0, sigma=100.0,
                        kinematic_group=1)

    for name in names:
        fit.set_bounds(feature=name, parameter='amplitude', bounds=[0.0, None])
        fit.set_bounds(feature=name + '_b', parameter='amplitude', bounds=[0.0, None])

    fit.add_minimize_constraint('n2_6548_b.sigma', '> n2_6548.sigma')
    fit.add_minimize_constraint('n2_6548.amplitude', 'n2_6583.amplitude / 3.06')
    fit.add_minimize_constraint('n2_6548_b.amplitude', 'n2_6583_b.amplitude / 3.06')

    return fit


def test_simple_fit():
    fit = simple_fit()
    fit.fit(verbose=True, fit_continuum=True)
    assert 1


def test_flux():
    fit = simple_fit()
    fit.fit(fit_continuum=True)
    fit.integrate_flux(sigma_factor=5.0)
    assert 1


def test_optimize_fit():
    fit = simple_fit()
    fit.optimize_fit(width=5.0)
    fit.fit(fit_continuum=True)
    assert 1


def test_good_fraction():
    fit = simple_fit()
    fit.flags[0:len(fit.flags):2] = True
    with pytest.raises(AssertionError) as err:
        fit.fit()
    assert 'valid pixels not reached' in str(err.value)


def test_skip_feature():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
    spec = onedspec.Spectrum(file_name)
    fit = modeling.LineFit(spec, fitting_window=(6400.0, 6700.0))
    with pytest.warns(UserWarning):
        fit.add_feature(name='hb', rest_wavelength=4861.0, amplitude=0, velocity=0, sigma=10)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.fit(fit_continuum=True)
    assert 1


def test_bounds():
    fit = simple_fit()
    fit.set_bounds('ha', 'velocity', [-50, 50])
    fit.set_bounds('ha', 'amplitude', [None, 0.4e-14])
    fit.fit(fit_continuum=True)
    assert 1


def test_constraints():
    fit = simple_fit()
    fit.add_minimize_constraint('n2_6548.sigma', 'n2_6583.sigma')
    fit.add_minimize_constraint('n2_6548.velocity', 'n2_6583.velocity')
    fit.add_minimize_constraint('n2_6548.amplitude', 'n2_6583.amplitude / 3.06')
    fit.add_minimize_constraint('n2_6548.amplitude', '< ha.amplitude')
    fit.add_minimize_constraint('ha.amplitude', '< 3.5e-15')
    fit.fit(fit_continuum=True)
    assert fit._get_feature_parameter('ha', 'amplitude', 'solution') < 3.6e-15


def test_gauss_hermite():
    fit = simple_fit(function='gauss_hermite')
    fit.add_minimize_constraint(parameter='n2_6548.h_3', expression='n2_6583.h_3')
    fit.add_minimize_constraint(parameter='n2_6548.h_4', expression='n2_6583.h_4')
    fit.fit(fit_continuum=True)
    assert 1


def test_kinematic_groups():
    fit = full_fit()
    fit.optimize_fit(width=5.0)
    fit.fit(fit_continuum=True)

    assert 1


def test_fixed_features():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
    spec = onedspec.Spectrum(file_name)
    fit = modeling.LineFit(spec, function='gaussian', fitting_window=(6400.0, 6800.0))
    names = ['n2_6548', 'ha', 'n2_6583', 's2_6716', 's2_6731']
    r_wl = [6548.04, 6562.8, 6583.46, 6716.44, 6730.86]

    for name, wl, k in zip(names, r_wl, [0, 1, 0, 2, 2]):
        fit.add_feature(name=name, rest_wavelength=wl, amplitude=1.0e-14, velocity=0.0, sigma=100.0,
                        fixed='n2' in name, kinematic_group=k)
    fit.fit(fit_continuum=True)
    assert 1


def test_monte_carlo():
    fit = full_fit()
    fit.optimize_fit(width=5.0)
    fit.fit(fit_continuum=True)
    fit.monte_carlo(3)

    assert 1


def test_simple_cube_fit():
    fit = simple_fit(fit_type='cube', individual_spec=(3, 4))
    fit.fit(fit_continuum=True)
    fit.plot(x_0=3, y_0=4)
    assert True


def test_spiral_fit():
    fit = simple_fit(fit_type='cube', spiral_fitting=True, spiral_center=(3, 4))
    fit.fit(fit_continuum=True)
    assert True
