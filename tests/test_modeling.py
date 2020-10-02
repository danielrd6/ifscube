import pkg_resources
import pytest

from ifscube import onedspec, modeling, datacube


def fit_select(function: str = 'gaussian', fit_type: str = 'spectrum', **kwargs):
    if fit_type == 'spectrum':
        file_name = pkg_resources.resource_filename('ifscube', 'examples/manga_onedspec.fits')
        spec = onedspec.Spectrum(file_name, primary='PRIMARY', scidata='F_OBS', variance='F_VAR', flags='F_FLAG',
                                 stellar='F_SYN')
        fit = modeling.LineFit(spec, function=function, fitting_window=(6400.0, 6800.0), **kwargs)
    elif fit_type == 'cube':
        file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc3081_cube.fits')
        cube = datacube.Cube(file_name, variance='ERR')
        fit = modeling.LineFit3D(cube, function=function, fitting_window=(6400.0, 6800.0), **kwargs)
    else:
        raise RuntimeError(f'fit_type "{fit_type}" not understood.')
    return fit


def simple_fit(function: str = 'gaussian', fit_type: str = 'spectrum', **kwargs):
    names = ['n2_6548', 'ha', 'n2_6583']
    r_wl = [6548.04, 6562.8, 6583.46]

    fit = fit_select(function, fit_type, **kwargs)
    if function == 'gauss_hermite':
        gh_moments = {'h_3': 0.0, 'h_4': 0.0}
    else:
        gh_moments = {}

    for name, wl in zip(names, r_wl):
        fit.add_feature(name=name, rest_wavelength=wl, amplitude=1.0, velocity=0.0, sigma=100.0,
                        kinematic_group=0, **gh_moments)

    for name in names:
        fit.set_bounds(feature=name, parameter='amplitude', bounds=[0.0, 100])
        fit.set_bounds(feature=name, parameter='sigma', bounds=[40.0, 300])
        fit.set_bounds(feature=name, parameter='velocity', bounds=[-300, 300])
        if function == 'gauss_hermite':
            fit.set_bounds(feature=name, parameter='h_3', bounds=[-0.2, 0.2])

    return fit


def full_fit(function: str = 'gaussian', fit_type: str = 'spectrum', **kwargs):
    fit = fit_select(function, fit_type, **kwargs)
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
        fit.set_bounds(feature=name, parameter='sigma', bounds=[40.0, 300])
        fit.set_bounds(feature=name, parameter='velocity', bounds=[-300, 300])
        fit.set_bounds(feature=name + '_b', parameter='amplitude', bounds=[0.0, None])

    fit.add_constraint('n2_6548_b.sigma', '> n2_6548.sigma')
    fit.add_constraint('n2_6548.amplitude', 'n2_6583.amplitude / 3.06')
    fit.add_constraint('n2_6548_b.amplitude', 'n2_6583_b.amplitude / 3.06')

    return fit


def test_simple_fit():
    fit = simple_fit()
    fit.fit(verbose=True, )
    assert 1


def test_flux():
    fit = simple_fit()
    fit.fit()
    fit.integrate_flux(sigma_factor=5.0)
    assert 1


def test_optimize_fit():
    fit = simple_fit()
    fit.optimize_fit(width=5.0)
    fit.fit()
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
    fit.fit()
    assert 1


def test_bounds():
    fit = simple_fit()
    fit.set_bounds('ha', 'velocity', [-50, 50])
    fit.set_bounds('ha', 'amplitude', [None, 100])
    fit.fit()
    assert 1


def test_constraints():
    fit = simple_fit()
    fit.add_constraint('n2_6548.amplitude', 'n2_6583.amplitude / 3.06')
    fit.add_constraint('n2_6548.amplitude', '< ha.amplitude')
    fit.add_constraint('ha.amplitude', '< 1.5')
    fit.fit()
    assert fit._get_feature_parameter('ha', 'amplitude', 'solution') < 1.6


def test_constraints_differential_evolution():
    fit = simple_fit()
    fit.add_constraint('n2_6548.amplitude', 'n2_6583.amplitude / 3.06')
    fit.add_constraint('ha.amplitude', '< 1.5')
    bounds = {'amplitude': [0, 10], 'velocity': [-300, 300], 'sigma': [40, 300]}
    for i, j in fit.parameter_names:
        fit.set_bounds(i, j, bounds[j])
    fit.fit(min_method='differential_evolution', verbose=True)
    assert True


def test_gauss_hermite():
    fit = simple_fit(function='gauss_hermite')
    fit.add_constraint('n2_6548.amplitude', 'n2_6583.amplitude / 3.06')
    fit.fit()
    assert 1


def test_kinematic_groups():
    fit = full_fit()
    fit.optimize_fit(width=5.0)
    fit.fit()
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
    fit.fit()
    assert 1


def test_monte_carlo():
    fit = full_fit()
    fit.optimize_fit(width=5.0)
    fit.fit()
    fit.monte_carlo(3)
    assert 1


def test_equivalent_width():
    fit = simple_fit(fit_type='spectrum')
    fit.optimize_fit(width=5.0)
    fit.fit()
    fit.equivalent_width()
    assert True


def test_velocity_width():
    fit = simple_fit(fit_type='spectrum')
    fit.optimize_fit(width=5.0)
    fit.fit()
    fit.velocity_width(feature='ha', width=80)
    assert True


def test_simple_cube_fit():
    fit = simple_fit(fit_type='cube')
    fit.fit()
    assert True


def test_full_cube_fit():
    fit = full_fit(fit_type='cube')
    fit.optimize_fit()
    fit.fit()
    assert True


def test_spiral_loop():
    fit = simple_fit(fit_type='cube', spiral_loop=True, spiral_center=(3, 4))
    fit.fit()
    assert True


def test_cube_monte_carlo():
    fit = fit_select(function='gaussian', fit_type='cube')
    names = ['n2_6548', 'ha', 'n2_6583']
    r_wl = [6548.04, 6562.8, 6583.46]

    for name, wl in zip(names, r_wl):
        fit.add_feature(name=name, rest_wavelength=wl, amplitude='peak', velocity=0.0, sigma=100.0,
                        kinematic_group=0)

    fit.optimize_fit()
    fit.fit()
    fit.monte_carlo(3)
    assert True


def test_cube_flux():
    fit = simple_fit(fit_type='cube', spiral_loop=True)
    fit.optimize_fit(width=5.0)
    fit.fit()
    fit.integrate_flux()
    assert True


def test_cube_equivalent_width():
    fit = simple_fit(fit_type='cube', spiral_loop=True)
    fit.optimize_fit(width=5.0)
    fit.fit()
    fit.equivalent_width()
    assert True


def test_refit():
    fit = simple_fit(fit_type='cube', spiral_loop=True, spiral_center=(3, 4), refit=True, refit_radius=2.0,
                     bounds_change=[0.2, 10, 10])
    fit.fit()
    assert True
