import copy
import warnings
from typing import Union, Iterable, Callable

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from astropy import constants, units
from scipy import integrate
from scipy.optimize import minimize, differential_evolution, LinearConstraint

from ifscube import parser, spectools
from ifscube import plots as ifs_plots
from ifscube import profiles
from .datacube import Cube
from .onedspec import Spectrum


class FixedSolution:
    def __init__(self, solution: np.ndarray):
        self.x = solution


class LineFit:
    def __init__(self, data: Union[Spectrum, Cube], function: str = 'gaussian', fitting_window: tuple = None,
                 instrument_dispersion: float = 1.0):
        self.monte_carlo_solutions = None
        self.input_data = data
        self.data = np.copy(data.data)
        self.stellar = np.copy(data.stellar)
        self.wavelength = np.copy(data.rest_wavelength)
        self.variance = np.copy(data.variance)
        self.flags = np.copy(data.flags)
        self.instrument_dispersion = instrument_dispersion
        self.model_spectrum = None

        self.mask = np.copy(self.flags)
        if fitting_window is None:
            fitting_window = self.wavelength[[0, -1]]
        self.fitting_window = fitting_window

        self.mask[(self.wavelength < fitting_window[0]) | (self.wavelength > fitting_window[1])] = True

        self.pseudo_continuum = np.zeros_like(self.data)
        self.weights = np.ones_like(self.data)

        self.bounds = []
        self.constraints = []

        if function == 'gaussian':
            self.function = profiles.gauss_vel
            self.parameters_per_feature = 3
        elif function == 'gauss_hermite':
            self.function = profiles.gauss_hermite_vel
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

        self.velocity_width_model = None
        self.velocity_width_direct = None

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
                a = self.data[~self.mask].max()
            elif amplitude == 'mean':
                a = self.data[~self.mask].mean()
            elif amplitude == 'median':
                a = np.median(self.data[~self.mask])
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
                    for key in [_ for _ in parameters.keys() if _ != 'amplitude']:
                        parameters[key] = self._get_feature_parameter(first_feature, key, 'initial_guess')
                else:
                    self.kinematic_groups[kinematic_group] = [name]
                    for i in parameter_names[1:]:
                        assert parameters[i] is not None, \
                            f'Parameter "{i}" for feature "{name}" is undefined. Since this is the first feature ' \
                            f'of kinematic group "{kinematic_group}", all of its kinematic parameters should be ' \
                            'defined.'

            self.feature_names.append(name)
            self.feature_wavelengths = np.append(self.feature_wavelengths, rest_wavelength)

            if type(parameters['amplitude']) == str:
                parameters['amplitude'] = self.amplitude_parser(parameters['amplitude'])

            self.initial_guess = np.append(self.initial_guess, [parameters[_] for _ in parameter_names])
            self.parameter_names += [(name, _) for _ in parameter_names]
            self.bounds += [[None, None]] * self.parameters_per_feature

    def set_bounds(self, feature: str, parameter: str, bounds: list):
        if (feature, parameter) in self.parameter_names:
            self.bounds[self.parameter_names.index((feature, parameter))] = bounds
        else:
            warnings.warn(f'Parameter {parameter} for feature {feature} not found. Attempt to set bounds failed.',
                          stacklevel=2)

    def add_constraint(self, parameter, expression):
        self.constraint_expressions.append([parameter, expression])

    def _evaluate_constraints(self, method):
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
                    cp.evaluate(parameter, scale_factor=self.flux_scale_factor, method=method)
                else:
                    cp.evaluate(parameter, method=method)
                constraints.append(cp.constraint)

        if method == 'differential_evolution':
            if len(constraints) > 0:
                c = {'A': np.array([_['matrix'] for _ in constraints]),
                     'lb': np.array([_['lower_bounds'] for _ in constraints]),
                     'ub': np.array([_['upper_bounds'] for _ in constraints])}
                for key in c.keys():
                    assert np.max(c[key].astype(bool).sum(0)) <= 1.0, \
                        f'Multiple constraints for the same parameter are not supported for the method "{method}".'
                    c[key] = c[key].sum(0)
                constraints = (LinearConstraint(**c),)
            else:
                constraints = ()
        self.constraints = constraints

    def res(self, x, s):
        x = self._unpack_parameters(x)
        m = self.function(self.wavelength[~self.mask], self.feature_wavelengths, x)
        a = np.square((s - m) * self.weights[~self.mask])
        b = a / self.variance[~self.mask]
        chi2 = np.sum(b)
        return chi2

    def pack_groups(self):
        if self.kinematic_groups != {}:
            features_in_groups = [item for sublist in list(self.kinematic_groups.values()) for item in sublist]
            for i in self.feature_names:
                if i not in features_in_groups:
                    last_key = max(self.kinematic_groups.keys()) + 1
                    self.kinematic_groups[last_key] = [i]

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
        if (self.flux_scale_factor <= 0.0) or (np.isnan(self.flux_scale_factor)):
            warnings.warn(f'Attempt to get flux scale factor failed with value {self.flux_scale_factor}. '
                          'Falling back to unity.', stacklevel=2)
            self.flux_scale_factor = 1.0
        for i in self._get_parameter_indices('amplitude'):
            self.initial_guess[i] /= self.flux_scale_factor
            self.bounds[i] = [_ / self.flux_scale_factor if _ is not None else _ for _ in self.bounds[i]]
        for i in ['data', 'pseudo_continuum', 'stellar']:
            setattr(self, i, getattr(self, i) / self.flux_scale_factor)
        if not np.all(self.variance == 1.0):
            self.variance /= np.square(self.flux_scale_factor)

    def _restore_flux_scale(self):
        for i in self._get_parameter_indices('amplitude'):
            self.initial_guess[i] *= self.flux_scale_factor
            self.solution[i] *= self.flux_scale_factor
            self.bounds[i] = [_ * self.flux_scale_factor if _ is not None else _ for _ in self.bounds[i]]
        for i in ['data', 'pseudo_continuum', 'stellar']:
            setattr(self, i, getattr(self, i) * self.flux_scale_factor)
        if not np.all(self.variance == 1.0):
            self.variance *= np.square(self.flux_scale_factor)

    def optimize_fit(self, width: float = 5.0):
        sigma = self.initial_guess[self._get_parameter_indices('sigma')]
        sigma_lam = spectools.sigma_lambda(sigma_vel=sigma, rest_wl=self.feature_wavelengths)
        optimized_mask = spectools.feature_mask(wavelength=self.wavelength, feature_wavelength=self.feature_wavelengths,
                                                sigma=sigma_lam, width=width)
        self.mask |= optimized_mask

    def fit_pseudo_continuum(self, degree: int = 6, line_weight: float = 1.0, weights=None,
                             n_iterate: int = 5, lower_threshold: float = 2.0, upper_threshold: float = 3):
        wl = self.wavelength
        fw = (wl >= self.fitting_window[0]) & (wl <= self.fitting_window[1])
        if weights is None:
            cw = np.ones_like(self.data)

            def continuum_weights(sigmas):
                for i, j in enumerate(self.feature_wavelengths):
                    low_lambda = j - (3.0 * sigmas[i])
                    up_lambda = j + (3.0 * sigmas[i])
                    cw[(wl > low_lambda) & (wl < up_lambda)] = line_weight
                return cw

            sigma_velocities = self._get_feature_parameter(feature='all', parameter='sigma', attribute='initial_guess')
            weights = continuum_weights(spectools.sigma_lambda(sigma_velocities, self.feature_wavelengths))
        # noinspection PyCallingNonCallable
        pc = spectools.continuum(wl[fw], (self.data - self.stellar)[fw], output='polynomial', degree=degree,
                                 n_iterate=n_iterate, lower_threshold=lower_threshold, upper_threshold=upper_threshold,
                                 weights=weights[fw])(wl)
        pc[~fw] = np.nan
        self.pseudo_continuum = pc

    def fit(self, min_method: str = 'slsqp', minimum_good_fraction: float = 0.8, verbose: bool = False, **kwargs):
        assert self.initial_guess.size > 0, 'There are no spectral features to fit. Aborting'
        assert self.initial_guess.size == (len(self.feature_names) * self.parameters_per_feature), \
            'There is a problem with the initial guess. Check the spectral feature definitions.'

        self.fit_arguments = {'min_method': min_method}
        self.fit_arguments.update(**kwargs)

        valid_fraction = np.sum(~self.mask & ~self.flags) / np.sum(~self.mask)
        assert valid_fraction >= minimum_good_fraction, 'Minimum fraction of valid pixels not reached: ' \
                                                        f'{valid_fraction} < {minimum_good_fraction}.'

        self._apply_flux_scale()

        self.pack_groups()
        p_0 = self._pack_parameters(self.initial_guess)
        bounds = self._pack_parameters(np.array(self.bounds))
        self._evaluate_constraints(min_method)
        s = (self.data - self.pseudo_continuum - self.stellar)[~self.mask & ~self.flags]

        if min_method in ['slsqp', 'trust-constr']:
            if ('options' not in kwargs) and (min_method == 'slsqp'):
                kwargs['options'] = {'eps': 1.e-3, 'ftol': 1e-8}
            # noinspection PyTypeChecker
            solution = minimize(self.res, x0=p_0, args=(s,), method=min_method, bounds=bounds,
                                constraints=self.constraints, **kwargs)
            self.status = solution.status
        elif min_method == 'differential_evolution':
            assert not any([_ is None for _ in np.ravel(bounds)]), \
                'Cannot have unbound parameters when using differential_evolution.'
            # noinspection PyTypeChecker
            solution = differential_evolution(
                self.res, bounds=list(bounds), args=(s,), constraints=self.constraints, **kwargs)
            self.status = 0
        elif min_method == 'fixed':
            solution = FixedSolution(p_0)
        else:
            raise RuntimeError(f'Unknown minimization method {min_method}.')
        self.reduced_chi_squared = self.res(solution.x, s) / self._degrees_of_freedom()

        self.solution = self._unpack_parameters(solution.x)
        self._restore_flux_scale()

        self.model_spectrum = (self.function(self.wavelength, self.feature_wavelengths, self.solution)
                               + self.pseudo_continuum + self.stellar)

        if verbose:
            print(self.print_parameters('solution'))

    def _degrees_of_freedom(self):
        dof = (np.sum(~self.mask) / self.instrument_dispersion) \
              - len(self._packed_parameter_names) - len(self.constraints) - 1
        return dof

    def print_parameters(self, attribute: str):
        if self._packed_parameter_names is None:
            s = []
        else:
            s = [f'\nReduced Chi^2: {self.reduced_chi_squared}',
                 f'Initially free parameters: {len(self._packed_parameter_names)}',
                 f'Number of constraints: {len(self.constraints)}', f'Valid data points: {np.sum(~self.mask)}',
                 f'Instrument dispersion: {self.instrument_dispersion} pixels',
                 f'Degrees of freedom: {self._degrees_of_freedom()}\n']
        p = getattr(self, attribute)
        u = getattr(self, 'uncertainties')
        for i, j in enumerate(self.parameter_names):
            if j[1] == 'amplitude':
                if u is None:
                    s.append(f'{j[0]}.{j[1]} = {p[i]:8.2e}')
                else:
                    s.append(f'{j[0]}.{j[1]} = {p[i]:8.2e} +- {u[i]:8.2e}')
            else:
                if u is None:
                    s.append(f'{j[0]}.{j[1]} = {p[i]:.2f}')
                else:
                    s.append(f'{j[0]}.{j[1]} = {p[i]:.2f} +- {u[i]:.2f}')
            if p[i] in self.bounds[i]:
                s[-1] = '\033[93m' + s[-1] + '\033[0m'
        s = '\n'.join(s)
        return s

    def monte_carlo(self, n_iterations: int = 10):
        assert self.solution is not None, 'Monte carlo uncertainties can only be evaluated after a successful fit.'
        assert not np.all(self.variance == 1.0), \
            'Cannot estimate uncertainties via Monte Carlo. The variance was not given.'
        old_data = copy.deepcopy(self.data)
        solution_matrix = np.zeros((n_iterations,) + self.solution.shape)
        self.uncertainties = np.zeros_like(self.solution)

        for c in range(n_iterations):
            self.data = np.random.normal(old_data, np.sqrt(self.variance))
            self.fit(**self.fit_arguments)
            solution_matrix[c] = copy.deepcopy(self.solution)
        self.solution = np.nanmean(solution_matrix, axis=0)
        self.uncertainties = np.nanstd(solution_matrix, axis=0)
        self.monte_carlo_solutions = solution_matrix
        self.data = old_data

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
        assert (feature in self.feature_names) or (feature == 'all'), \
            f'Requested feature "{feature}" not defined. Currently defined features are: {self.feature_names}'
        x = getattr(self, attribute)
        if parameter == 'all':
            i = self.parameter_names.index((feature, 'amplitude'))
            return x[i:i + self.parameters_per_feature]
        elif feature == 'all':
            idx = self._get_parameter_indices(parameter=parameter)
            return x[idx]
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

        for idx, feature in enumerate(self.feature_names):
            if np.any(np.isnan(self._get_feature_parameter(feature, 'all', 'solution'))):
                flux_model[idx] = np.nan
                flux_direct[idx] = np.nan

            else:
                z = (1.0 + self._get_feature_parameter(feature, 'velocity', 'solution') / c)
                center_wavelength = np.array([self.feature_wavelengths[idx] * z])
                sigma_vel = np.array([self._get_feature_parameter(feature, 'sigma', 'solution')])
                sigma_lam = spectools.sigma_lambda(sigma_vel, center_wavelength)
                mask = spectools.feature_mask(wavelength=self.wavelength, feature_wavelength=center_wavelength,
                                              sigma=sigma_lam, width=sigma_factor)

                observed_feature = self.data[~mask] - self.pseudo_continuum[~mask] - self.stellar[~mask]
                fit = self.function(self.wavelength[~mask], np.array([self.feature_wavelengths[idx]]),
                                    self._get_feature_parameter(feature, 'all', 'solution'))
                for i, j in enumerate(self.feature_names):
                    if j != feature:
                        observed_feature -= self.function(
                            self.wavelength[~mask], np.array([self.feature_wavelengths[i]]),
                            self._get_feature_parameter(j, 'all', 'solution'))

                flux_model[idx] = integrate.trapezoid(fit, x=self.wavelength[~mask])
                flux_direct[idx] = integrate.trapezoid(observed_feature, x=self.wavelength[~mask])

            self.flux_model = flux_model
            self.flux_direct = flux_direct

    def velocity_width(self, feature: Union[str, list], sigma_factor: float = 5.0, plot: bool = False, **kwargs):
        """
        Evaluates the velocity width which encompasses a given percentile
        of the spectral feature's flux.

        Parameters
        ----------
        feature : str, list
            Names of the features to be included in the modelled
            spectrum. Can be either a string specifying a single
            spectral feature, or a list of strings.
        sigma_factor : float
            Width, in units of the feature's sigma parameter, to
            be considered when integrating the line flux.
        plot : bool
            Plots results.
        **kwargs
            To be passed to spectools.velocity_width

        Returns
        -------
        res : dict
            Dictionary containing the results from
            spectools.velocity_width.
        """
        fit = LineFit.get_model(self, feature=feature)

        if "rest_wavelength" not in kwargs:
            if type(feature) == str:
                rwl = self.feature_wavelengths[self.feature_names.index(feature)]
            elif type(feature) == list:
                rwl = np.average([self.feature_wavelengths[self.feature_names.index(_)] for _ in feature])
            else:
                raise TypeError('feature must be either string or list.')
            kwargs["rest_wavelength"] = rwl

        obs_spec = (self.data - self.pseudo_continuum - self.stellar)
        res = spectools.velocity_width(wavelength=self.wavelength, model=fit, data=obs_spec, sigma_factor=sigma_factor,
                                       **kwargs)

        if plot and not np.isnan(res['model_velocity_width']):
            ifs_plots.velocity_width(res)

        self.velocity_width_model = res['model_velocity_width']
        self.velocity_width_direct = res['direct_velocity_width']

        return res

    def get_model(self, feature: Union[str, list]):
        """
        Generates an array, equal in length to the original spectrum,
        but having only the spectral features defined by *feature*.

        Parameters
        ----------
        feature : str, list
            Names of the features to be included in the modelled
            spectrum. Can be either a string specifying a single
            spectral feature, or a list of strings.

        Returns
        -------
        fit : np.ndarray
            Modelled spectrum.
        """
        if type(feature) == str:
            index = self.feature_names.index(feature)
            solution = self._get_feature_parameter(feature, 'all', 'solution')
            fit = self.function(self.wavelength, np.array([self.feature_wavelengths[index]]), solution)
        elif type(feature) == list:
            fit = np.zeros_like(self.wavelength)
            for component in feature:
                solution = self._get_feature_parameter(component, 'all', 'solution')
                index = self.feature_names.index(component)
                fit += self.function(self.wavelength, np.array([self.feature_wavelengths[index]]), solution)
        else:
            raise TypeError('feature must be either string or list.')
        return fit

    def _get_center_wavelength(self, feature):
        velocity = self._get_feature_parameter(feature, 'velocity', 'solution')
        f_wl = self.feature_wavelengths[self.feature_names.index(feature)]
        lam = (velocity * units.km / units.s).to(
            units.angstrom, equivalencies=units.doppler_relativistic(f_wl * units.angstrom))
        return lam.value

    def _cut_fit_spectrum(self, feature, sigma_factor):
        sigma = self._get_feature_parameter(feature, 'sigma', 'solution')
        solution = self._get_feature_parameter(feature, 'all', 'solution')
        wavelength = self.feature_wavelengths[self.feature_names.index(feature)]
        sig = spectools.sigma_lambda(sigma, wavelength)
        low_wl, up_wl = [self._get_center_wavelength(feature) + (_ * sigma_factor * sig) for _ in [-1.0, 1.0]]
        cond = (self.wavelength > low_wl) & (self.wavelength < up_wl)
        fit = self.function(self.wavelength[cond], np.array([wavelength]), solution)

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
                    mask = (
                            ((self.wavelength > cwin[0]) & (self.wavelength < cwin[1]))
                            | ((self.wavelength > cwin[2]) & (self.wavelength < cwin[3])))
                    n_iterations = 1
                    degree = 1
                else:
                    mask = ((self.wavelength > self.fitting_window[0]) & (self.wavelength < self.fitting_window[1]))
                    n_iterations = 3
                    degree = 7

                # noinspection PyCallingNonCallable
                cont = spectools.continuum(
                    self.wavelength[mask], (syn + fit_continuum)[mask], degree=degree, n_iterate=n_iterations,
                    lower_threshold=3, upper_threshold=3, output='polynomial')(self.wavelength[cond])

                # Remember that 1 - (g + c)/c = -g/c, where g is the
                # line profile and c is the local continuum level.
                #
                # That is why we can shorten the equivalent width
                # definition in the eqw_model integration below.
                ci = component_index
                if np.all(cont == 0.0):
                    warnings.warn(f'Continuum for feature {component} is identically zero. Setting EqW to NaN.')
                    eqw_model[ci] = np.nan
                    eqw_direct[ci] = np.nan
                else:
                    eqw_model[ci] = integrate.trapezoid(- fit / cont, x=self.wavelength[cond])
                    eqw_direct[ci] = integrate.trapezoid(1. - (data[cond] / cont), x=self.wavelength[cond])

            self.eqw_model = eqw_model
            self.eqw_direct = eqw_direct

    def plot(self, figure: plt.Figure = None, plot_all: bool = True, verbose: bool = False,
             return_results: bool = False, spectrum_ax: plt.axes = None, residuals_ax: plt.axes = None):
        if plot_all:
            m = Ellipsis
        else:
            m = ~self.mask
        wavelength = self.wavelength[m]
        observed = self.data[m]
        pseudo_continuum = self.pseudo_continuum[m]
        stellar = self.stellar[m]
        solution = np.array(self.solution, dtype="float64")
        model_lines = self.function(wavelength, self.feature_wavelengths, solution)
        err = np.sqrt(self.variance[m])

        if figure is None:
            fig = plt.figure()
        else:
            fig = figure

        if (spectrum_ax is None) and (residuals_ax is None):
            fig.clf()
            spectrum_ax, residuals_ax = fig.subplots(nrows=2, ncols=1, sharex='all',
                                                     gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0})

        residuals_ax.plot(wavelength, (observed - model_lines - stellar - pseudo_continuum) / err)
        residuals_ax.set_xlabel('Wavelength')
        residuals_ax.set_ylabel('Residuals / sigma')
        residuals_ax.grid()

        if self.uncertainties is not None:
            low = self.function(wavelength, self.feature_wavelengths, solution - self.uncertainties)
            high = self.function(wavelength, self.feature_wavelengths, solution + self.uncertainties)
            low += stellar + pseudo_continuum
            high += stellar + pseudo_continuum
            spectrum_ax.fill_between(wavelength, low, high, color='C2', alpha=0.5)

        spectrum_ax.plot(wavelength, observed, color='C0')
        spectrum_ax.plot(wavelength, model_lines + stellar + pseudo_continuum, color='C2')

        if np.any(pseudo_continuum):
            spectrum_ax.plot(wavelength, pseudo_continuum + stellar)
        if np.any(stellar):
            spectrum_ax.plot(wavelength, stellar)

        ppf = self.parameters_per_feature
        for i in range(0, len(self.parameter_names), ppf):
            feature_wl = np.array([self.feature_wavelengths[int(i / ppf)]])
            parameters = solution[i:i + ppf]
            line = self.function(wavelength, feature_wl, parameters)
            spectrum_ax.plot(wavelength, pseudo_continuum + stellar + line, 'k--')

        spectrum_ax.set_ylabel('Spectral flux density')
        spectrum_ax.minorticks_on()

        spectrum_ax.grid()

        if verbose:
            print(self.print_parameters('solution'))
        elif return_results:
            return self.print_parameters('solution')


class LineFit3D(LineFit):
    two_d_attributes = ['reduced_chi_squared', 'status']

    def __init__(self, data: Cube, function: str = 'gaussian', fitting_window: tuple = None,
                 instrument_dispersion: float = 1.0, individual_spec: tuple = None, spiral_loop: bool = False,
                 spiral_center: Union[tuple, str] = None, refit: bool = False, refit_radius: float = 3.0,
                 bounds_change: list = None):
        super().__init__(data, function, fitting_window, instrument_dispersion)
        self.refit = refit
        self.refit_radius = refit_radius
        if refit:
            assert bounds_change is not None, 'When using refit the bounds_change option must be defined.'
        self.bounds_change = bounds_change

        if individual_spec is not None:
            self.spaxel_indices = np.array([individual_spec[::-1]])
        elif spiral_loop:
            self._spiral(spiral_center)
        else:
            self.spaxel_indices = data.spec_indices
        self.y_0, self.x_0 = self.spaxel_indices[0]
        self.status = np.ones(self.data.data.shape[1:])
        self.image_coordinates = np.indices(self.data.data.shape[1:])

    def _spiral(self, spiral_center: Union[str, tuple] = None):
        y, x = self.input_data.spec_indices[:, 0], self.input_data.spec_indices[:, 1]
        if spiral_center is None:
            spiral_center = (x.max() / 2., y.max() / 2.)
        elif isinstance(spiral_center, str):
            summed = np.nan_to_num(np.sum(self.data, 0))
            if spiral_center == 'peak':
                spiral_center = np.where(summed == summed.max())
            elif spiral_center == 'cofm':
                from scipy.ndimage import center_of_mass as com
                spiral_center = np.array(com(summed)).round().astype(int)
            else:
                raise ValueError('If spiral_center is a string, it must be either "peak" or "cofm".')
        elif isinstance(spiral_center, tuple):
            pass
        else:
            raise TypeError('Parameter spiral_center must be either tuple or string.')

        radius = np.sqrt(np.square((x - spiral_center[0])) + np.square((y - spiral_center[1])))
        angle = np.arctan2(y - spiral_center[1], x - spiral_center[0])
        angle[angle < 0] += 2 * np.pi
        b = np.array([(np.ravel(radius)[i], np.ravel(angle)[i]) for i in range(len(np.ravel(radius)))],
                     dtype=[('radius', 'f8'), ('angle', 'f8')])
        s = np.argsort(b, axis=0, order=['radius', 'angle'])
        self.spaxel_indices = np.column_stack([np.ravel(y)[s], np.ravel(x)[s]])

    def optimize_fit(self, width: float = 5.0):
        sigma = self.initial_guess[self._get_parameter_indices('sigma')]
        sigma_lam = spectools.sigma_lambda(sigma_vel=sigma, rest_wl=self.feature_wavelengths)
        optimized_mask = spectools.feature_mask(wavelength=self.wavelength, feature_wavelength=self.feature_wavelengths,
                                                sigma=sigma_lam, width=width)
        self.mask |= optimized_mask[:, np.newaxis, np.newaxis]

    def _mask2d_to_nan(self, attribute):
        x = getattr(self, attribute)
        if len(x.shape) == 3:
            x[:, self.input_data.spatial_mask] = np.nan
        elif len(x.shape) == 2:
            x[self.input_data.spatial_mask] = np.nan

    def fit_pseudo_continuum(self, degree: int = 6, line_weight: float = 1.0, weights=None,
                             n_iterate: int = 5, lower_threshold: float = 2.0, upper_threshold: float = 3):
        attributes = ['data', 'variance', 'flags', 'mask', 'pseudo_continuum', 'stellar', 'weights', 'status']
        mod_attr = ['pseudo_continuum']
        self.pseudo_continuum = np.zeros_like(self.data)
        self._mask2d_to_nan('pseudo_continuum')
        self._select_spaxel(function=super().fit_pseudo_continuum, used_attributes=attributes,
                            modified_attributes=mod_attr,
                            kwargs={'degree': degree, 'line_weight': line_weight, 'weights': weights,
                                    'n_iterate': n_iterate, 'lower_threshold': lower_threshold,
                                    'upper_threshold': upper_threshold},
                            description='Fitting pseudo continuum')

    def fit(self, min_method: str = 'slsqp', minimum_good_fraction: float = 0.8, verbose: bool = False,
            refit: bool = False, refit_radius: float = 3.0, bounds_change: float = 0.2, **kwargs):
        if self.solution is None:
            self.solution = np.zeros((len(self.initial_guess),) + self.data.shape[1:])
        if self.reduced_chi_squared is None:
            self.reduced_chi_squared = np.zeros(self.data.shape[1:])
        if self.model_spectrum is None:
            self.model_spectrum = np.zeros_like(self.data)
        if self.status is None:
            self.status = np.ones(self.data.shape[1:])
        for i in ['solution', 'reduced_chi_squared', 'model_spectrum']:
            self._mask2d_to_nan(i)

        attributes = ['data', 'variance', 'flags', 'mask', 'pseudo_continuum', 'stellar', 'weights', 'status',
                      'solution', 'reduced_chi_squared', 'model_spectrum']
        mod_attr = ['solution', 'reduced_chi_squared', 'status', 'model_spectrum']
        fit_args = (min_method, minimum_good_fraction, verbose)
        self._select_spaxel(function=super().fit, used_attributes=attributes, modified_attributes=mod_attr,
                            args=fit_args, kwargs=kwargs, description='Fitting spectral features', fitting=True)

    def integrate_flux(self, sigma_factor: float = 5.0):
        attributes = ['data', 'variance', 'flags', 'mask', 'pseudo_continuum', 'stellar', 'weights', 'status',
                      'solution', 'reduced_chi_squared', 'flux_model', 'flux_direct']
        in_attr = ['flux_model', 'flux_direct']
        dims = (len(self.feature_names),) + self.data.shape[1:]
        self.flux_direct = np.zeros(dims)
        self.flux_model = np.zeros(dims)
        for i in ['flux_model', 'flux_direct']:
            self._mask2d_to_nan(i)
        self._select_spaxel(function=super().integrate_flux, used_attributes=attributes, modified_attributes=in_attr,
                            args=(sigma_factor,), description='Integrating flux')

    def equivalent_width(self, sigma_factor=5.0, continuum_windows=None):
        attributes = ['data', 'variance', 'flags', 'mask', 'pseudo_continuum', 'stellar', 'weights', 'status',
                      'solution', 'reduced_chi_squared', 'eqw_model', 'eqw_direct']
        in_attr = ['eqw_model', 'eqw_direct']
        dims = (len(self.feature_names),) + self.data.shape[1:]
        self.eqw_direct = np.zeros(dims)
        self.eqw_model = np.zeros(dims)
        for i in ['eqw_model', 'eqw_direct']:
            self._mask2d_to_nan(i)
        self._select_spaxel(function=super().equivalent_width, used_attributes=attributes, modified_attributes=in_attr,
                            args=(sigma_factor, continuum_windows), description='Evaluating equivalent width')

    def velocity_width(self, feature: Union[str, list], sigma_factor=5.0, **kwargs):
        in_attr = ['velocity_width_model', 'velocity_width_direct']
        attributes = ['data', 'flags', 'mask', 'pseudo_continuum', 'stellar', 'status', 'solution'] + in_attr
        self.velocity_width_model = np.zeros(self.data.shape[1:])
        self.velocity_width_direct = np.zeros(self.data.shape[1:])
        for i in ['velocity_width_model', 'velocity_width_direct']:
            self._mask2d_to_nan(i)
        r = self._select_spaxel(function=super().velocity_width, used_attributes=attributes,
                                modified_attributes=in_attr, args=(feature, sigma_factor), kwargs=kwargs,
                                description='Evaluating velocity width')
        return r

    def get_model(self, feature: Union[str, list]):
        attributes = ['data', 'flags', 'mask', 'pseudo_continuum', 'stellar', 'status', 'solution']
        r = self._select_spaxel(function=super().get_model, used_attributes=attributes,
                                modified_attributes=None, args=(feature,), description='Making modelled data cube.')
        x = np.zeros_like(self.data)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                x[:, i, j] = r[i, j]
        return x

    @staticmethod
    def _change_bounds(value, change, original, fraction: bool = False):
        if fraction:
            lb, ub = [value * (1.0 + (_ * np.sign(value) * change)) for _ in [-1, 1]]
        else:
            lb, ub = [value + (_ * change) for _ in [-1, 1]]

        if original[0] is not None:
            if lb < original[0]:
                lb = original[0]
        else:
            lb = None
        if original[1] is not None:
            if ub > original[1]:
                ub = original[1]
        else:
            ub = None
        if (lb is not None) and (ub is not None):
            assert lb < ub, 'Lower bound higher than upper bound.'
        return [lb, ub]

    def _updated_parameters(self, x: int, y: int, cube_solution: np.ndarray, cube_status: np.ndarray,
                            original_bounds: list, original_guess: np.ndarray):
        yy, xx = self.image_coordinates
        r = np.sqrt(np.square(yy - y) + np.square(np.square(xx - x)))
        mask = (r < self.refit_radius) & (cube_status == 0)
        if np.any(mask):
            self.initial_guess = np.nanmean(cube_solution[:, mask], axis=1)
            b = []
            for i in range(len(self.bounds)):
                d = (i % self.parameters_per_feature)
                b.append(self._change_bounds(self.initial_guess[i], self.bounds_change[d], original_bounds[i],
                                             fraction=(d == 0)))
            self.bounds = b
        else:
            self.initial_guess = original_guess.copy()
            self.bounds = original_bounds.copy()

    def _select_spaxel(self, function: Callable, x: Union[int, Iterable] = None, y: Union[int, Iterable] = None,
                       used_attributes: list = None, modified_attributes: list = None, description: str = None,
                       args: tuple = None, kwargs: dict = None, fitting: bool = False):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        if modified_attributes is None:
            modified_attributes = []

        cube_data = {_: np.copy(getattr(self, _)) for _ in used_attributes}
        r = np.empty(self.data.shape[1:], dtype='object')

        refit = self.refit and fitting
        if refit:
            original_guess = self.initial_guess.copy()
            original_bounds = self.bounds.copy()
            if (self.bounds_change is not None) and hasattr(self.bounds_change, '__len__'):
                assert len(self.bounds_change) == self.parameters_per_feature, \
                    'Bounds_change should have the same length as feature parameters. ' \
                    f'Expected {self.parameters_per_feature}, got {len(self.bounds_change)}.'
            else:
                raise RuntimeError('Bounds_change should be an iterable.')

        if (x is None) and (y is None):
            x = self.spaxel_indices[:, 1]
            y = self.spaxel_indices[:, 0]
        if isinstance(x, (int, np.int64)):
            iterator = [[x, y]]
        else:
            iterator = tqdm.tqdm(np.column_stack([x, y]), desc=description, unit='spaxels')
        for xx, yy in iterator:
            s = (Ellipsis, yy, xx)
            for i in used_attributes:
                if i in self.two_d_attributes:
                    setattr(self, i, cube_data[i][s[1:]])
                else:
                    setattr(self, i, cube_data[i][s])

            if refit:
                # noinspection PyUnboundLocalVariable
                self._updated_parameters(xx, yy, cube_solution=cube_data['solution'], cube_status=cube_data['status'],
                                         original_bounds=original_bounds, original_guess=original_guess)
            try:
                r[s[1:]] = function(*args, **kwargs)
            except Exception as err:
                warnings.warn(f"Exception occurred in spaxel {s[1:]}.\n" + str(err), category=RuntimeWarning,
                              stacklevel=2)
                self.status = 1

            for i in modified_attributes:
                if i in self.two_d_attributes:
                    cube_data[i][s[1:]] = getattr(self, i)
                else:
                    cube_data[i][s] = getattr(self, i)

        for i in used_attributes:
            setattr(self, i, cube_data[i])

        return r

    def plot(self, figure: plt.Figure = None, plot_all: bool = True, verbose: bool = False,
             return_results: bool = False, x_0: int = None, y_0: int = None, spectrum_ax: plt.axes = None,
             residuals_ax: plt.axes = None):
        attributes = ['data', 'variance', 'flags', 'mask', 'pseudo_continuum', 'stellar', 'solution',
                      'reduced_chi_squared']
        if self.uncertainties is not None:
            attributes.append('uncertainties')
        if x_0 is None:
            x_0 = self.x_0
        if y_0 is None:
            y_0 = self.y_0
        r = self._select_spaxel(function=super().plot, x=x_0, y=y_0, used_attributes=attributes,
                                args=(figure, plot_all, verbose, return_results, spectrum_ax, residuals_ax))
        return r
