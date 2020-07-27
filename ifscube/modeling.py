import copy
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants
from scipy import integrate
from scipy.optimize import minimize, differential_evolution

from ifscube import elprofile, parser, spectools
from .onedspec import Spectrum


class LineFit:
    def __init__(self, spectrum: Spectrum, function: str = 'gaussian', fitting_window: tuple = None,
                 fit_continuum: bool = False, continuum_options: dict = None):
        self.fitting_window = fitting_window

        self.data = spectrum.data
        self.stellar = spectrum.stellar
        self.wavelength = spectrum.rest_wavelength
        self.variance = spectrum.variance
        self.flags = spectrum.flags

        self.mask = np.zeros_like(self.data, dtype=bool)
        if fitting_window is not None:
            self.mask[(self.wavelength < fitting_window[0]) | (self.wavelength > fitting_window[1])] = True

        self.flux_scale_factor = np.nanmedian(self.data[~self.mask])
        self.data /= self.flux_scale_factor
        self.stellar /= self.flux_scale_factor
        if not np.all(self.variance == 1.0):
            self.variance /= self.flux_scale_factor ** 2

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

        self.fit_status = 1
        self.fit_arguments = None

        self.flux_model = np.array([])
        self.flux_direct = np.array([])

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
            if fixed:
                self.fixed_features.append(name)
            if kinematic_group is not None:
                if kinematic_group in self.kinematic_groups.keys():
                    self.kinematic_groups[kinematic_group].append(name)
                    first_feature = self.kinematic_groups[kinematic_group][0]
                    velocity = self._get_feature_parameter(first_feature, 'velocity', 'initial_guess')
                    sigma = self._get_feature_parameter(first_feature, 'sigma', 'initial_guess')
                    if self.parameters_per_feature == 5:
                        h_3 = self._get_feature_parameter(first_feature, 'h_3', 'initial_guess')
                        h_4 = self._get_feature_parameter(first_feature, 'h_4', 'initial_guess')
                else:
                    self.kinematic_groups[kinematic_group] = [name]
            else:
                assert self.kinematic_groups == {}, \
                    'If you plan on using k_group, please set it for every spectral feature. ' \
                    'Groups can have a single spectral feature in them.'

            self.feature_names.append(name)
            self.feature_wavelengths = np.concatenate([self.feature_wavelengths, [rest_wavelength]])

            if type(amplitude) == str:
                amplitude = self.amplitude_parser(amplitude)

            if self.parameters_per_feature == 3:
                self.initial_guess = np.concatenate(
                    [self.initial_guess, [amplitude / self.flux_scale_factor, velocity, sigma]])
            elif self.parameters_per_feature == 5:
                self.initial_guess = np.concatenate(
                    [self.initial_guess, [amplitude / self.flux_scale_factor, velocity, sigma, h_3, h_4]])

            self.parameter_names += [
                (name, _) for _ in ['amplitude', 'velocity', 'sigma', 'h_3', 'h_4'][:self.parameters_per_feature]]
            self.bounds += [[None, None]] * self.parameters_per_feature

    def set_bounds(self, feature: str, parameter: str, bounds: list):
        if parameter == 'amplitude':
            bounds = [_ / self.flux_scale_factor if _ is not None else _ for _ in bounds]
        self.bounds[self.parameter_names.index((feature, parameter))] = bounds

    def add_minimize_constraint(self, parameter, expression):
        self.constraint_expressions.append([parameter, expression])

    def _evaluate_constraints(self):
        pn = ['.'.join(_) for _ in self._packed_parameter_names]
        constraints = []
        for parameter, expression in self.constraint_expressions:
            warn_msg = f'Constraint on parameter "{parameter}" is being ignored because this feature is set to fixed.'
            if parameter.split('.')[0] in self.fixed_features: warnings.warn(warn_msg)
            else:
                cp = parser.ConstraintParser(expr=expression, parameter_names=pn)
                cp.evaluate(parameter)
                constraints.append(cp.constraint)

        return constraints

    def res(self, x, s):
        x = self._unpack_parameters(x)
        m = self.function(self.wavelength[~self.mask], self.feature_wavelengths, x)
        a = np.square((s - m) * self.weights[~self.mask])
        b = a / self.variance[~self.mask]
        rms = np.sqrt(np.sum(b))
        return rms

    def _pack_groups(self):
        packed = []
        first_in_group = [self.kinematic_groups[_][0] for _ in self.kinematic_groups.keys()]
        inverse = []
        fixed_factor = []

        for j, i in enumerate(self.parameter_names):
            name, parameter = i
            if name not in self.fixed_features:
                if parameter == 'amplitude':
                    packed.append(i)
                    inverse.append(packed.index(i))
                else:
                    if self.kinematic_groups != {}:
                        if name in first_in_group:
                            packed.append(i)
                            inverse.append(packed.index(i))
                        else:
                            for key in self.kinematic_groups.keys():
                                group = self.kinematic_groups[key]
                                if name in group:
                                    inverse.append(packed.index((group[0], parameter)))
                    else:
                        packed.append(i)
                        inverse.append(packed.index(i))
            else:
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
                            self.initial_guess[j]
                            / self._get_feature_parameter(self.fixed_features[0], 'amplitude', 'initial_guess'))
                    inverse.append(None)

        unfixed_inverse_transform = [i for i, j in enumerate(inverse) if j is not None]
        inverse_transform = np.array([_ for _ in inverse if _ is not None])

        fixed_factor = np.array(fixed_factor)
        transform = np.array([self.parameter_names.index(_) for _ in packed])
        fixed_amplitude_indices = np.array([self._get_parameter_indices('amplitude', _) for _ in self.fixed_features])

        if self.fixed_features:
            def pack_parameters(x):
                return x[transform]

            def unpack_parameters(x):
                new_x = np.copy(self.initial_guess)
                new_x[unfixed_inverse_transform] = x[inverse_transform]
                new_x[fixed_amplitude_indices] = [new_x[first_fixed_index] * _ for _ in fixed_factor]
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

        self._pack_groups()
        p_0 = self._pack_parameters(self.initial_guess)
        bounds = self._pack_parameters(np.array(self.bounds))
        constraints = self._evaluate_constraints()

        s = self.data[~self.mask] - self.pseudo_continuum[~self.mask] - self.stellar[~self.mask]
        if min_method == 'slsqp':
            # noinspection PyTypeChecker
            solution = minimize(self.res, x0=p_0, args=(s,), method=min_method, bounds=bounds, constraints=constraints,
                                options=minimize_options)
            self.solution = self._unpack_parameters(solution.x)
        elif min_method == 'differential_evolution':
            assert not any([_ is None for _ in np.ravel(bounds)]), \
                'Cannot have unbound parameters when using differential_evolution.'
            self.solution = self._unpack_parameters(
                differential_evolution(self.res, bounds=bounds, args=(s,)).x)
        else:
            raise RuntimeError(f'Unknown minimization method {min_method}.')

        if verbose:
            self.print_parameters('solution')

    def print_parameters(self, attribute: str):
        p = getattr(self, attribute)
        u = getattr(self, 'uncertainties')
        for i, j in enumerate(self.parameter_names):
            if j[1] == 'amplitude':
                if u is None:
                    print(f'{j[0]}.{j[1]} = {p[i] * self.flux_scale_factor:8.2e}')
                else:
                    print(f'{j[0]}.{j[1]} = {p[i] * self.flux_scale_factor:8.2e} +- '
                          f'{u[i] * self.flux_scale_factor:8.2e}')
            else:
                if u is None:
                    print(f'{j[0]}.{j[1]} = {p[i]:.2f}')
                else:
                    print(f'{j[0]}.{j[1]} = {p[i]:.2f} +- {u[i]:.2f}')

    def monte_carlo(self, n_iterations: int = 10, verbose: bool = False):
        assert self.solution is not None, 'Monte carlo uncertainties can only be evaluated after a successful fit.'
        p = self.solution
        old_data = copy.deepcopy(self.data)
        solution_matrix = np.zeros((n_iterations, p.size))
        self.uncertainties = np.zeros_like(p)

        c = 0
        while c < n_iterations:
            self.data = np.random.normal(old_data, np.sqrt(self.variance))
            self.fit(**self.fit_arguments, verbose=False)
            solution_matrix[c] = copy.deepcopy(self.solution)
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

            self.flux_model = flux_model * self.flux_scale_factor
            self.flux_direct = flux_direct * self.flux_scale_factor

    def plot(self, plot_all: bool = False):
        sf = self.flux_scale_factor
        wavelength = self.wavelength[~self.mask]
        observed = (self.data * self.flux_scale_factor)[~self.mask]
        pseudo_continuum = self.pseudo_continuum[~self.mask] * sf
        stellar = self.stellar[~self.mask] * sf
        model = self.function(wavelength, self.feature_wavelengths, self.solution) * sf

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if self.uncertainties is not None:
            low = self.function(wavelength, self.feature_wavelengths, self.solution - self.uncertainties) * sf
            high = self.function(wavelength, self.feature_wavelengths, self.solution + self.uncertainties) * sf
            low += stellar + pseudo_continuum
            high += stellar + pseudo_continuum
            ax.fill_between(wavelength, low, high, color='C2', alpha=0.5)

        ax.plot(wavelength, observed, color='C0')
        ax.plot(wavelength, model + stellar + pseudo_continuum, color='C2')

        if plot_all:
            ax.plot(wavelength, pseudo_continuum)
            if np.any(stellar):
                ax.plot(wavelength, stellar)
            ax.plot(wavelength, observed - model - stellar - pseudo_continuum)

            ppf = self.parameters_per_feature
            for i in range(0, len(self.parameter_names), ppf):
                feature_wl = self.feature_wavelengths[int(i / ppf)]
                parameters = self.solution[i:i + ppf]
                line = self.function(wavelength, feature_wl, parameters) * self.flux_scale_factor
                ax.plot(wavelength, pseudo_continuum + stellar + line, 'k--')

        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Spectral flux density')
        ax.minorticks_on()

        ax.grid()
        plt.show()
