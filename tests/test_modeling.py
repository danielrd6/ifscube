import numpy as np
import importlib.resources
import pytest
from astropy import units

from ifscube import onedspec, modeling, datacube


def fit_select(function: str = 'gaussian', fit_type: str = 'spectrum', **kwargs):
    if fit_type == 'spectrum':
        file_name = importlib.resources.files("ifscube") / 'examples/manga_onedspec.fits'
        spec = onedspec.Spectrum(file_name, primary='PRIMARY', scidata='F_OBS', variance='F_VAR', flags='F_FLAG',
                                 stellar='F_SYN')
        fit = modeling.LineFit(spec, function=function, fitting_window=(6400.0, 6800.0), **kwargs)
    elif fit_type == 'cube':
        file_name = importlib.resources.files("ifscube") / 'examples/ngc3081_cube.fits'
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

    if fit_type == 'spectrum':
        amp = 1.0
    elif fit_type == 'cube':
        amp = 1.0e-14
    else:
        raise IOError(f'fit_type {fit_type} is not recognized.')

    for name, wl in zip(names, r_wl):
        fit.add_feature(name=name, rest_wavelength=wl, amplitude=amp, velocity=0.0, sigma=100.0,
                        kinematic_group=0, **gh_moments)

    for name in names:
        fit.set_bounds(feature=name, parameter='amplitude', bounds=[0.0, amp * 100])
        fit.set_bounds(feature=name, parameter='sigma', bounds=[40.0, 300])
        fit.set_bounds(feature=name, parameter='velocity', bounds=[-300, 300])
        if function == 'gauss_hermite':
            fit.set_bounds(feature=name, parameter='h_3', bounds=[-0.2, 0.2])

    fit.fit_pseudo_continuum()
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

    fit.fit_pseudo_continuum()
    return fit


def test_peak_amplitude():
    names = ['n2_6548', 'ha', 'n2_6583']
    r_wl = [6548.04, 6562.8, 6583.46]

    fit = fit_select(function='gaussian', fit_type='spectrum')

    for name, wl in zip(names, r_wl):
        fit.add_feature(name=name, rest_wavelength=wl, amplitude='peak', velocity=0.0, sigma=100.0, kinematic_group=0)
    fit.fit()
    assert fit.initial_guess[0] == 4.147481615680209


def test_mean_amplitude():
    names = ['n2_6548', 'ha', 'n2_6583']
    r_wl = [6548.04, 6562.8, 6583.46]

    fit = fit_select(function='gaussian', fit_type='spectrum')

    for name, wl in zip(names, r_wl):
        fit.add_feature(name=name, rest_wavelength=wl, amplitude='mean', velocity=0.0, sigma=100.0, kinematic_group=0)
    fit.fit()
    assert fit.initial_guess[0] == 2.4758099883101328


def test_median_amplitude():
    names = ['n2_6548', 'ha', 'n2_6583']
    r_wl = [6548.04, 6562.8, 6583.46]

    fit = fit_select(function='gaussian', fit_type='spectrum')

    for name, wl in zip(names, r_wl):
        fit.add_feature(name=name, rest_wavelength=wl, amplitude='median', velocity=0.0, sigma=100.0, kinematic_group=0)
    fit.fit()
    assert fit.initial_guess[0] == 2.40130144892458


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
    file_name = importlib.resources.files("ifscube") / 'examples/ngc6300_nuc.fits'
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
    ratio = fit._get_feature_parameter("n2_6583", "amplitude", "solution") \
        / fit._get_feature_parameter("n2_6548", "amplitude", "solution")
    constraint_a = np.abs(1 - (ratio / 3.06)) < 0.01
    constraint_b = fit._get_feature_parameter("ha", "amplitude", "solution") < 1.5
    assert constraint_a and constraint_b


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
    file_name = importlib.resources.files("ifscube") / 'examples/ngc6300_nuc.fits'
    spec = onedspec.Spectrum(file_name)
    fit = modeling.LineFit(spec, function='gaussian', fitting_window=(6400.0, 6800.0))
    names = ['n2_6548', 'ha', 'n2_6583', 's2_6716', 's2_6731']
    r_wl = [6548.04, 6562.8, 6583.46, 6716.44, 6730.86]

    for name, wl, k in zip(names, r_wl, [0, 1, 0, 2, 2]):
        amp = 1e-14 if name == 'n2_6548' else 3.e-14
        fit.add_feature(name=name, rest_wavelength=wl, amplitude=amp, velocity=0.0, sigma=100.0,
                        fixed='n2' in name, kinematic_group=k)
    fit.fit()
    assert 1


def test_get_model():
    fit = simple_fit()
    fit.fit()
    a = fit.get_model("ha")
    index = fit.feature_names.index("ha")
    solution = fit._get_feature_parameter("ha", "all", "solution")
    b = fit.function(fit.wavelength, index, solution)
    assert np.all(a - b) == 0


def test_get_model_3d():
    fit = simple_fit(fit_type="cube")
    fit.fit()
    fit.get_model("ha")
    assert True


def test_get_model_multiple():
    fit = simple_fit()
    fit.fit()
    a = fit.get_model(["ha", "n2_6583"])

    index = fit.feature_names.index("ha")
    solution = fit._get_feature_parameter("ha", "all", "solution")
    b = fit.function(fit.wavelength, index, solution)

    index = fit.feature_names.index("n2_6583")
    solution = fit._get_feature_parameter("n2_6583", "all", "solution")
    b += fit.function(fit.wavelength, index, solution)

    assert np.all(a - b) == 0


def test_get_model_multiple_3d():
    fit = simple_fit(fit_type="cube")
    fit.fit()
    fit.get_model(["ha", "n2_6583"])
    assert True


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
    fit.velocity_width(feature='ha', width=80, rest_wavelength=units.Quantity(6562.8, 'angstrom'))
    assert True


def test_velocity_width_multiple_features():
    fit = simple_fit(fit_type='spectrum')
    fit.optimize_fit(width=5.0)
    fit.fit()
    fit.velocity_width(feature=['n2_6548', 'ha', 'n2_6583'], width=80,
                       rest_wavelength=units.Quantity(6562.8, 'angstrom'))
    assert True


def test_simple_cube_fit():
    fit = simple_fit(fit_type='cube')
    fit.fit()
    assert True


def test_simple_cube_warning():
    """
    Tests if the exception in spaxel warning is issued correctly.
    """
    fit = simple_fit(fit_type='cube')
    fit.fit()
    with pytest.warns(RuntimeWarning, match="Exception occurred in spaxel"):
        fit.velocity_width(feature=['ha', 'n2_6583'], width=80)


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


def test_cube_velocity_width_multiple_features():
    fit = simple_fit(fit_type='cube')
    fit.optimize_fit(width=5.0)
    fit.fit()
    fit.velocity_width(feature=['ha', 'n2_6583'], width=80, rest_wavelength=units.Quantity(6572.9, 'Angstrom'))
    assert True


def test_cube_velocity_width_multiple_features_not_fractional():
    fit = simple_fit(fit_type='cube')
    fit.optimize_fit(width=5.0)
    fit.fit()
    fit.velocity_width(feature=['ha', 'n2_6583'], width=80, rest_wavelength=units.Quantity(6572.9, 'Angstrom'),
                       fractional_pixels=False)
    assert True


def test_refit():
    fit = simple_fit(fit_type='cube', spiral_loop=True, spiral_center=(3, 4), refit=True, refit_radius=2.0,
                     bounds_change=[0.2, 10, 10])
    fit.fit()
    assert True


def test_spiral_center():
    file_name = importlib.resources.files("ifscube") / 'examples/ngc3081_cube.fits'
    cube = datacube.Cube(file_name)
    fit = modeling.LineFit3D(data=cube, spiral_center=None, spiral_loop=True)
    assert all(fit.spaxel_indices[0] == [4, 3]) and all(fit.spaxel_indices[-1] == [0, 5])
    fit = modeling.LineFit3D(data=cube, spiral_center=(3, 4), spiral_loop=True)
    assert all(fit.spaxel_indices[0] == [4, 3]) and all(fit.spaxel_indices[-1] == [0, 0])
    fit = modeling.LineFit3D(data=cube, spiral_center='peak', spiral_loop=True)
    assert all(fit.spaxel_indices[0] == [2, 4]) and all(fit.spaxel_indices[-1] == [7, 0])
    fit = modeling.LineFit3D(data=cube, spiral_center='cofm', spiral_loop=True)
    assert all(fit.spaxel_indices[0] == [2, 4]) and all(fit.spaxel_indices[-1] == [7, 0])
    with pytest.raises(ValueError):
        modeling.LineFit3D(data=cube, spiral_center='boo', spiral_loop=True)
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        modeling.LineFit3D(data=cube, spiral_center=['boo'], spiral_loop=True)


def test_missing_k_group():
    file_name = importlib.resources.files("ifscube") / 'examples/manga_onedspec.fits'
    spec = onedspec.Spectrum(file_name, primary='PRIMARY', scidata='F_OBS', variance='F_VAR', flags='F_FLAG',
                             stellar='F_SYN')
    fit = modeling.LineFit(spec, function='gaussian', fitting_window=(6400.0, 6800.0))
    fit.add_feature(name='line_a', rest_wavelength=6548, amplitude=1, velocity=0.0, sigma=100.0)
    fit.add_feature(name='line_b', rest_wavelength=6563, amplitude=1, velocity=0.0, sigma=100.0, kinematic_group=1)
    fit.pack_groups()
    assert True
