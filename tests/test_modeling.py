import pkg_resources
import pytest

from ifscube import onedspec, modeling


def setup_fit(function: str = 'gaussian'):
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
    spec = onedspec.Spectrum(file_name)
    fit = modeling.LineFit(spec, function=function, fitting_window=(6400.0, 6700.0), fit_continuum=True)
    fit.add_feature(name='n2_6548', rest_wavelength=6548.04, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='n2_6583', rest_wavelength=6583.46, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    return fit


def setup_full_fit(function: str = 'gaussian'):
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
    spec = onedspec.Spectrum(file_name)
    fit = modeling.LineFit(spec, function=function, fitting_window=(6400.0, 6800.0), fit_continuum=True)
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
    fit = setup_fit()
    fit.fit(verbose=True)
    assert 1


def test_flux():
    fit = setup_fit()
    fit.fit()
    fit.integrate_flux(sigma_factor=5.0)
    print(fit.flux_model)
    print(fit.flux_direct)
    assert 1


def test_optimize_fit():
    fit = setup_fit()
    fit.optimize_fit(width=5.0)
    fit.fit()
    assert 1


def test_good_fraction():
    fit = setup_fit()
    fit.flags[0:len(fit.flags):2] = True
    with pytest.raises(AssertionError) as err:
        fit.fit()
    assert 'valid pixels not reached' in str(err.value)


def test_skip_feature():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
    spec = onedspec.Spectrum(file_name)
    fit = modeling.LineFit(spec, fitting_window=(6400.0, 6700.0), fit_continuum=True)
    with pytest.warns(UserWarning):
        fit.add_feature(name='hb', rest_wavelength=4861.0, amplitude=0, velocity=0, sigma=10)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.fit()
    assert 1


def test_bounds():
    fit = setup_fit()
    fit.set_bounds('ha', 'velocity', [-50, 50])
    fit.set_bounds('ha', 'amplitude', [None, 0.4e-14])
    fit.fit()
    assert 1


def test_constraints():
    fit = setup_fit()
    fit.add_minimize_constraint('n2_6548.sigma', 'n2_6583.sigma')
    fit.add_minimize_constraint('n2_6548.velocity', 'n2_6583.velocity')
    fit.add_minimize_constraint('n2_6548.amplitude', 'n2_6583.amplitude / 3.06')
    fit.add_minimize_constraint('n2_6548.amplitude', '< ha.amplitude')
    fit.fit()
    assert 1


def test_gauss_hermite():
    fit = setup_fit(function='gauss_hermite')
    fit.add_minimize_constraint(parameter='n2_6548.h_3', expression='n2_6583.h_3')
    fit.add_minimize_constraint(parameter='n2_6548.h_4', expression='n2_6583.h_4')
    fit.fit()
    assert 1


def test_kinematic_groups():
    fit = setup_full_fit()
    fit.optimize_fit(width=5.0)
    fit.fit()

    assert 1


def test_monte_carlo():
    fit = setup_full_fit()
    fit.optimize_fit(width=5.0)
    fit.fit()
    fit.monte_carlo(10)

    assert 1