import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants, units
from scipy import integrate
from scipy.optimize import minimize, differential_evolution

from ifscube import elprofile, parser, spectools
from .onedspec import Spectrum


class LineFit:
    def __init__(self, spectrum: Spectrum, function: str = 'gaussian', fitting_window: tuple = None,
                 fit_continuum: bool = False, continuum_options: dict = None, instrument_dispersion: float = 1.0):
        self.fitting_window = fitting_window

        self.data = spectrum.data
        self.stellar = spectrum.stellar
        self.wavelength = spectrum.rest_wavelength
        self.variance = spectrum.variance
        self.flags = spectrum.flags
        self.instrument_dispersion = instrument_dispersion
        self.model_spectrum = None

        self.mask = np.copy(self.flags)
        if fitting_window is not None:
            self.mask[(self.wavelength < fitting_window[0]) | (self.wavelength > fitting_window[1])] = True

        if fit_continuum:
            if continuum_options is None:
                continuum_options = {}
            self.pseudo_continuum = spectools.continuum(
                self.wavelength, (self.data - self.stellar), output='function', **continuum_options)[1]
        else:
            self.pseudo_continuum = np.zeros_like(self.data)

        self.weights = np.ones_like(self.data)

        self.bounds = []
        self.constraints = []

        if function == 'gaussian':
            self.function = elprofile.gaussvel
            self.parameters_per_feature = 3
        elif function == 'gauss_hermite':
            self.function = elprofile.gausshermitevel
            self.parameters_per_feature = 5

        self.solution = None
        self.uncertainties = None

        self.feature_names = []
        self.parameter_names = []
        self.initial_guess = np.array([])
        self.feature_wavelengths = np.array([])
        self.constraint_expressions = []

        self.kinematic_groups = {}
        self.fixed_features = []
        self._pack_parameters = None
        self._unpack_parameters = None
        self._packed_parameter_names = None

        self.status = 1
        self.fit_arguments = None
        self.reduced_chi_squared = None

        self.flux_model = np.array([])
        self.flux_direct = np.array([])

        self.eqw_model = None
        self.eqw_direct = None

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value: np.ndarray):
        assert value.size == self.data.size, 'Weights must be of the same size as data.'
        self._weights = value

    @property
    def flags(self):
        return self._flags

    @flags.setter
    def flags(self, value: np.ndarray):
        assert value.size == self.data.size, 'Flags must be of the same size as data.'
        assert value.dtype == bool, 'Flags must be an array of boolean type.'
        self._flags = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value: np.ndarray):
        assert value.size == self.data.size, 'Mask must be of the same size as data.'
        assert value.dtype == bool, 'Mask must be an array of boolean type.'
        self._mask = value

    def amplitude_parser(self, amplitude):
        try:
            if amplitude == 'peak':
                a = self.data[self.mask].max()
            elif amplitude == 'mean':
                a = self.data[self.mask].mean()
            elif amplitude == 'median':
                a = np.median(self.data[self.mask])
            else:
                raise ValueError(f'Amplitude {amplitude} not understood.')
        except ValueError:
            a = np.nan

        return a

    def add_feature(self, name: str, rest_wavelength: float, amplitude: Union[float, str], velocity: float = None,
                    sigma: float = None, h_3: float = None, h_4: float = None, kinematic_group: int = None,
                    fixed: bool = False):

        if (rest_wavelength < self.fitting_window[0]) or (rest_wavelength > self.fitting_window[1]):
            warnings.warn(f'Spectral feature {name} outside of fitting window. Skipping.')
        else:
            loc = locals()
            parameter_names = ['amplitude', 'velocity', 'sigma', 'h_3', 'h_4'][:self.parameters_per_feature]
            parameters = {_: loc[_] for _ in parameter_names}
            if fixed:
                self.fixed_features.append(name)
            if kinematic_group is not None:
                if kinematic_group in self.kinematic_groups.keys():
                    self.kinematic_groups[kinematic_group].append(name)
                    first_feature = self.kinematic_groups[kinematic_group][0]
                    for key in parameters.keys():
                        parameters[key] = self._get_feature_parameter(first_feature, key, 'initial_guess')
                else:
                    self.kinematic_groups[kinematic_group] = [name]
            else:
                assert self.kinematic_groups == {}, \
                    'If you plan on using k_group, please set it for every spectral feature. ' \
                    'Groups can have a single spectral feature in them.'

            self.feature_names.append(name)
            self.feature_wavelengths = np.append(self.feature_wavelengths, rest_wavelength)

            if type(parameters['amplitude']) == str:
                parameters['amplitude'] = self.amplitude_parser(parameters['amplitude'])

            self.initial_guess = np.append(self.initial_guess, [parameters[_] for _ in parameter_names])
            self.parameter_names += [(name, _) for _ in parameter_names]
            self.bounds += [[None, None]] * self.parameters_per_feature

    def set_bounds(self, feature: str, parameter: str, bounds: list):
        self.bounds[self.parameter_names.index((feature, parameter))] = bounds

    def add_minimize_constraint(self, parameter, expression):
        self.constraint_expressions.append([parameter, expression])

    def _evaluate_constraints(self):
        pn = ['.'.join(_) for _ in self._packed_parameter_names]
        constraints = []
        for parameter, expression in self.constraint_expressions:
            if parameter.split('.')[0] in self.fixed_features:
                warnings.warn(
                    f'Constraint on parameter "{parameter}" is being ignored because this feature is set to fixed.',
                    category=RuntimeWarning)
            else:
                cp = parser.ConstraintParser(expr=expression, parameter_names=pn)
                if 'amplitude' in parameter:
                    cp.evaluate(parameter, self.flux_scale_factor)
                else:
                    cp.evaluate(parameter)
                constraints.append(cp.constraint)
        self.constraints = constraints

    def res(self, x, s):
        x = self._unpack_parameters(x)
        m = self.function(self.wavelength[~self.mask], self.feature_wavelengths, x)
        a = np.square((s - m) * self.weights[~self.mask])
        b = a / self.variance[~self.mask]
        chi2 = np.sum(b)
        return chi2

    def _pack_groups(self):
        packed = []
        first_in_group = [_[0] for _ in self.kinematic_groups.values()]
        inverse = []
        fixed_factor = []

        for j, i in enumerate(self.parameter_names):
            name, parameter = i
            if name in self.fixed_features:
                if name == self.fixed_features[0]:
                    if parameter == 'amplitude':
                        packed.append(i)
                        first_fixed_index = packed.index(i)
                        inverse.append(first_fixed_index)
                        fixed_factor.append(1.0)
                    else:
                        inverse.append(None)
                else:
                    if parameter == 'amplitude':
                        fixed_factor.append(
                            self.initial_guess[j] / self._get_feature_parameter(
                                self.fixed_features[0], 'amplitude', 'initial_guess'))
                    inverse.append(None)
            else:
                if parameter == 'amplitude':
                    packed.append(i)
                    inverse.append(packed.index(i))
                else:
                    if self.kinematic_groups != {}:
                        if name in first_in_group:
                            packed.append(i)
                            inverse.append(packed.index(i))
                        else:
                            group = [_ for _ in self.kinematic_groups.values() if name in _][0]
                            assert group[0] not in self.fixed_features, \
                                'The first feature in a kinematic group should not be set to fixed. ' \
                                f'Check your definition of the "{group[0]}" feature.'
                            inverse.append(packed.index((group[0], parameter)))
                    else:
                        packed.append(i)
                        inverse.append(packed.index(i))

        free_parameter_loc = [i for i, j in enumerate(inverse) if j is not None]
        inverse_transform = [_ for _ in inverse if _ is not None]

        fixed_factor = np.array(fixed_factor)
        transform = np.array([self.parameter_names.index(_) for _ in packed])
        fixed_amplitude_indices = np.array([self._get_parameter_indices('amplitude', _) for _ in self.fixed_features])

        if self.fixed_features:
            new_x = np.copy(self.initial_guess)

            def pack_parameters(x):
                return x[transform]

            def unpack_parameters(x):
                new_x[free_parameter_loc] = x[inverse_transform]
                new_x[fixed_amplitude_indices] = [x[first_fixed_index] * _ for _ in fixed_factor]
                return new_x
        else:
            def pack_parameters(x):
                return x[transform]

            def unpack_parameters(x):
                return x[inverse_transform]

        self._pack_parameters = pack_parameters
        self._unpack_parameters = unpack_parameters
        self._packed_parameter_names = packed

    def _get_parameter_indices(self, parameter: str, feature: str = None):
        if feature is None:
            return np.array([self.parameter_names.index(_) for _ in self.parameter_names if parameter in _])
        else:
            return self.parameter_names.index((feature, parameter))

    def _apply_flux_scale(self):
        self.flux_scale_factor = np.percentile(np.abs(self.data[~self.mask]), 90)
        for i in self._get_parameter_indices('amplitude'):
            self.initial_guess[i] /= self.flux_scale_factor
            self.bounds[i] = [_ / self.flux_scale_factor if _ is not None else _ for _ in self.bounds[i]]
        for i in ['data', 'pseudo_continuum', 'stellar']:
            setattr(self, i, getattr(self, i) / self.flux_scale_factor)

    def _restore_flux_scale(self):
        for i in self._get_parameter_indices('amplitude'):
            self.initial_guess[i] *= self.flux_scale_factor
            self.solution[i] *= self.flux_scale_factor
            self.bounds[i] = [_ / self.flux_scale_factor if _ is not None else _ for _ in self.bounds[i]]
        for i in ['data', 'pseudo_continuum', 'stellar']:
            setattr(self, i, getattr(self, i) * self.flux_scale_factor)

    def optimize_fit(self, width: float = 5.0):
        sigma = self.initial_guess[self._get_parameter_indices('sigma')]
        sigma_lam = spectools.sigma_lambda(sigma_vel=sigma, rest_wl=self.feature_wavelengths)
        optimized_mask = spectools.feature_mask(wavelength=self.wavelength, feature_wavelength=self.feature_wavelengths,
                                                sigma=sigma_lam, width=width)
        self.mask |= optimized_mask

    def fit(self, min_method: str = 'slsqp', minimize_options: dict = None, minimum_good_fraction: float = 0.8,
            verbose: bool = False):
        assert self.initial_guess.size > 0, 'There are no spectral features to fit. Aborting'
        assert self.initial_guess.size == (len(self.feature_names) * self.parameters_per_feature), \
            'There is a problem with the initial guess. Check the spectral feature definitions.'

        self.fit_arguments = {'min_method': min_method, 'minimize_options': minimize_options}
        if minimize_options is None:
            minimize_options = {'eps': 1.0e-3}

        valid_fraction = np.sum(~self.mask & ~self.flags) / np.sum(~self.mask)
        assert valid_fraction >= minimum_good_fraction, 'Minimum fraction of valid pixels not reached: ' \
                                                        f'{valid_fraction} < {minimum_good_fraction}.'

        self._apply_flux_scale()

        self._pack_groups()
        p_0 = self._pack_parameters(self.initial_guess)
        bounds = self._pack_parameters(np.array(self.bounds))
        self._evaluate_constraints()
        s = (self.data - self.pseudo_continuum - self.stellar)[~self.mask & ~self.flags]

        if min_method == 'slsqp':
            # noinspection PyTypeChecker
            solution = minimize(self.res, x0=p_0, args=(s,), method=min_method, bounds=bounds,
                                constraints=self.constraints, options=minimize_options)
            self.status = solution.status
        elif min_method == 'differential_evolution':
            assert not any([_ is None for _ in np.ravel(bounds)]), \
                'Cannot have unbound parameters when using differential_evolution.'
            if self.constraints:
                warnings.warn('Constraints are not yet implemented for the "differential_evolution" method.',
                              category=RuntimeWarning)
            solution = differential_evolution(self.res, bounds=bounds, args=(s,))
        else:
            raise RuntimeError(f'Unknown minimization method {min_method}.')
        self.reduced_chi_squared = self.res(solution.x, s) / self._degrees_of_freedom()

        self.solution = self._unpack_parameters(solution.x)
        self._restore_flux_scale()

        self.model_spectrum = (self.function(self.wavelength, self.feature_wavelengths, self.solution)
                               + self.pseudo_continuum + self.stellar)

        if verbose:
            self.print_parameters('solution')

    def _degrees_of_freedom(self):
        dof = (np.sum(~self.mask) / self.instrument_dispersion) \
              - len(self._packed_parameter_names) - len(self.constraints) - 1
        return dof

    def print_parameters(self, attribute: str):
        print(f'\nReduced Chi^2: {self.reduced_chi_squared}')
        print(f'Initially free parameters: {len(self._packed_parameter_names)}')
        print(f'Number of constraints: {len(self.constraints)}')
        print(f'Valid data points: {np.sum(~self.mask)}')
        print(f'Instrument dispersion: {self.instrument_dispersion} pixels')
        print(f'Degrees of freedom: {self._degrees_of_freedom()}\n\n')
        p = getattr(self, attribute)
        u = getattr(self, 'uncertainties')
        for i, j in enumerate(self.parameter_names):
            if j[1] == 'amplitude':
                if u is None:
                    print(f'{j[0]}.{j[1]} = {p[i]:8.2e}')
                else:
                    print(f'{j[0]}.{j[1]} = {p[i]:8.2e} +- {u[i]:8.2e}')
            else:
                if u is None:
                    print(f'{j[0]}.{j[1]} = {p[i]:.2f}')
                else:
                    print(f'{j[0]}.{j[1]} = {p[i]:.2f} +- {u[i]:.2f}')

    def monte_carlo(self, n_iterations: int = 10, verbose: bool = False):
        assert self.solution is not None, 'Monte carlo uncertainties can only be evaluated after a successful fit.'
        old_data = np.copy(self.data)
        solution_matrix = np.zeros((n_iterations, self.solution.size))
        self.uncertainties = np.zeros_like(self.solution)

        c = 0
        while c < n_iterations:
            self.data = np.random.normal(old_data, np.sqrt(self.variance))
            self.fit(**self.fit_arguments, verbose=False)
            solution_matrix[c] = np.copy(self.solution)
            c += 1
        self.solution = solution_matrix.mean(axis=0)
        self.uncertainties = solution_matrix.std(axis=0)
        self.data = old_data

        if verbose:
            self.print_parameters('solution')

    def _get_feature_parameter(self, feature: str, parameter: str, attribute: str):
        """
        Access parameters by name and attribute.

        Parameters
        ----------
        feature : str
            Feature name
        parameter : str
            Parameter name (e.g. amplitude, velocity).
        attribute : str
            Attribute (e.g. solution, initial_guess).

        Returns
        -------

        """
        x = getattr(self, attribute)
        if parameter == 'all':
            i = self.parameter_names.index((feature, 'amplitude'))
            return x[i:i + self.parameters_per_feature]
        else:
            return x[self.parameter_names.index((feature, parameter))]

    def integrate_flux(self, sigma_factor: float = 5.0):
        """
        Integrates the flux of spectral features.

        Parameters
        ----------
        sigma_factor: float
            Radius of integration as a number of line sigmas.

        Returns
        -------
        None.
        """

        flux_model = np.zeros((len(self.feature_names),))
        flux_direct = np.zeros_like(flux_model)
        c = constants.c.to('km / s').value

        for idx, component in enumerate(self.feature_names):
            if np.any(np.isnan(self._get_feature_parameter(component, 'all', 'solution'))):
                flux_model[idx] = np.nan
                flux_direct[idx] = np.nan

            else:
                z = (1.0 + self._get_feature_parameter(component, 'velocity', 'solution') / c)
                center_wavelength = np.array([self.feature_wavelengths[idx] * z])
                sigma_vel = np.array([self._get_feature_parameter(component, 'sigma', 'solution')])
                sigma_lam = spectools.sigma_lambda(sigma_vel, center_wavelength)
                mask = spectools.feature_mask(wavelength=self.wavelength, feature_wavelength=center_wavelength,
                                              sigma=sigma_lam, width=sigma_factor)

                observed_feature = self.data[~mask] - self.pseudo_continuum[~mask] - self.stellar[~mask]
                fit = self.function(self.wavelength[~mask], self.feature_wavelengths[idx],
                                    self._get_feature_parameter(component, 'all', 'solution'))
                for i, j in enumerate(self.feature_names):
                    if j != component:
                        observed_feature -= self.function(
                            self.wavelength[~mask], self.feature_wavelengths[i],
                            self._get_feature_parameter(j, 'all', 'solution'))

                flux_model[idx] = integrate.trapz(fit, x=self.wavelength[~mask])
                flux_direct[idx] = integrate.trapz(observed_feature, x=self.wavelength[~mask])

            self.flux_model = flux_model
            self.flux_direct = flux_direct

    def _get_center_wavelength(self, feature):
        velocity = self._get_feature_parameter(feature, 'velocity', 'solution')
        f_wl = self.feature_wavelengths[self.feature_names.index(feature)]
        lam = (velocity * units.km / units.s).to(
            units.angstrom, equivalencies=units.doppler_relativistic(f_wl * units.angstrom))
        return lam.value

    def _cut_fit_spectrum(self, feature, sigma_factor):
        sigma = self._get_feature_parameter(feature, 'sigma', 'solution')
        index = self._get_parameter_indices('amplitude', feature)
        feature_solution = self._get_feature_parameter(feature, 'all', 'solution')
        sig = spectools.sigma_lambda(sigma, index)
        low_wl, up_wl = [self._get_center_wavelength(feature) + (_ * sigma_factor * sig) for _ in [-1.0, 1.0]]
        cond = (self.wavelength > low_wl) & (self.wavelength < up_wl)
        fit = self.function(self.wavelength[cond], np.array([self.feature_wavelengths[index]]), feature_solution)

        return cond, fit, self.stellar, self.pseudo_continuum, self.data

    def equivalent_width(self, sigma_factor=5.0, continuum_windows=None):
        """
        Evaluates the equivalent width of a previous fit.

        Parameters
        ----------
        sigma_factor: float
            Radius of integration as a number of line sigmas.
        continuum_windows: numpy.ndarray
          Continuum fitting windows in the form
          [blue0, blue1, red0, red1].

        Returns
        -------
        eqw : numpy.ndarray
          Equivalent widths measured on the emission line model and
          directly on the observed spectrum, respectively.
        """

        eqw_model = np.zeros((len(self.feature_names),))
        eqw_direct = np.zeros_like(eqw_model)

        for component in self.feature_names:

            component_index = self.feature_names.index(component)
            nan_data = ~np.isfinite(self._get_feature_parameter(component, 'amplitude', 'solution'))
            null_center_wavelength = ~np.isfinite(self._get_feature_parameter(component, 'velocity', 'solution'))

            if nan_data or null_center_wavelength:
                eqw_model[component_index] = np.nan
                eqw_direct[component_index] = np.nan
            else:
                cond, fit, syn, fit_continuum, data = self._cut_fit_spectrum(component, sigma_factor)

                # If the continuum fitting windows are set, use that
                # to define the weights vector.
                if continuum_windows is not None:
                    if component in continuum_windows:
                        cwin = continuum_windows[component]
                    else:
                        cwin = None
                else:
                    cwin = None

                if cwin is not None:
                    assert len(cwin) == 4, 'Windows must be an iterable of the form (blue0, blue1, red0, red1)'
                    weights = np.zeros_like(self.wavelength)
                    cwin_cond = (
                            ((self.wavelength > cwin[0]) & (self.wavelength < cwin[1]))
                            | ((self.wavelength > cwin[2]) & (self.wavelength < cwin[3])))
                    weights[cwin_cond] = 1
                    nite = 1
                else:
                    weights = np.ones_like(self.wavelength)
                    nite = 3

                cont = spectools.continuum(
                    self.wavelength, syn + fit_continuum, weights=weights, degree=1, n_iterate=nite, lower_threshold=3,
                    upper_threshold=3, output='function')[1][cond]

                # Remember that 1 - (g + c)/c = -g/c, where g is the
                # line profile and c is the local continuum level.
                #
                # That is why we can shorten the equivalent width
                # definition in the eqw_model integration below.
                ci = component_index
                eqw_model[ci] = integrate.trapz(- fit / cont, x=self.wavelength[cond])
                eqw_direct[ci] = integrate.trapz(1. - data[cond] / cont, x=self.wavelength[cond])

            self.eqw_model = eqw_model
            self.eqw_direct = eqw_direct

    def plot(self, plot_all: bool = True):
        wavelength = self.wavelength[~self.mask]
        observed = self.data[~self.mask]
        pseudo_continuum = self.pseudo_continuum[~self.mask]
        stellar = self.stellar[~self.mask]
        model_lines = self.function(wavelength, self.feature_wavelengths, self.solution)

        fig = plt.figure()

        if plot_all:
            ax, ax_res = fig.subplots(nrows=2, ncols=1, sharex='all',
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})
            ax_res.plot(wavelength, observed - model_lines - stellar - pseudo_continuum)
            ax_res.set_xlabel('Wavelength')
            ax_res.set_ylabel('Residuals')
            ax_res.grid()
        else:
            ax = fig.add_subplot(111)
            ax.set_xlabel('Wavelength')

        if self.uncertainties is not None:
            low = self.function(wavelength, self.feature_wavelengths, self.solution - self.uncertainties)
            high = self.function(wavelength, self.feature_wavelengths, self.solution + self.uncertainties)
            low += stellar + pseudo_continuum
            high += stellar + pseudo_continuum
            ax.fill_between(wavelength, low, high, color='C2', alpha=0.5)

        ax.plot(wavelength, observed, color='C0')
        ax.plot(wavelength, model_lines + stellar + pseudo_continuum, color='C2')

        if plot_all:
            if np.any(pseudo_continuum):
                ax.plot(wavelength, pseudo_continuum + stellar)
            if np.any(stellar):
                ax.plot(wavelength, stellar)

            ppf = self.parameters_per_feature
            for i in range(0, len(self.parameter_names), ppf):
                feature_wl = self.feature_wavelengths[int(i / ppf)]
                parameters = self.solution[i:i + ppf]
                line = self.function(wavelength, feature_wl, parameters)
                ax.plot(wavelength, pseudo_continuum + stellar + line, 'k--')

        ax.set_ylabel('Spectral flux density')
        ax.minorticks_on()

        ax.grid()
        plt.show()
