import copy
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants
from scipy import integrate
from scipy.optimize import minimize

from ifscube import elprofile, parser, spectools
from .onedspec import Spectrum


class LineFit:
    def __init__(self, spectrum: Spectrum, function: str = 'gaussian', fitting_window: tuple = None,
                 fit_continuum: bool = False):
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
            self.pseudo_continuum = spectools.continuum(
                self.wavelength, (self.data - self.stellar), output='function')[1]
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
        self.kinematic_group_transform = None
        self.kinematic_group_inverse_transform = None

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
        except ValueError:
            a = np.nan

        return a

    def add_feature(self, name: str, rest_wavelength: float, amplitude: Union[float, str], velocity: float = 0.0,
                    sigma: float = 100.0, h_3: float = 0.0, h_4: float = 0.0, kinematic_group: int = None):

        if (rest_wavelength < self.fitting_window[0]) or (rest_wavelength > self.fitting_window[1]):
            warnings.warn(f'Spectral feature {name} outside of fitting window. Skipping.')
        else:
            if kinematic_group is not None:
                if kinematic_group in self.kinematic_groups.keys():
                    self.kinematic_groups[kinematic_group].append(name)
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
        if self.kinematic_groups != {}:
            pn = [self.parameter_names[_] for _ in self.kinematic_group_transform]
        else:
            pn = self.parameter_names

        pn = ['.'.join(_) for _ in pn]
        constraints = []
        for parameter, expression in self.constraint_expressions:
            cp = parser.ConstraintParser(expr=expression, parameter_names=pn)
            cp.evaluate(parameter)
            constraints.append(cp.constraint)

        return constraints

    def res(self, x, s):
        x = x[self.kinematic_group_inverse_transform]
        m = self.function(self.wavelength[~self.mask], self.feature_wavelengths, x)
        a = np.square((s - m) * self.weights[~self.mask])
        b = a / self.variance[~self.mask]
        rms = np.sqrt(np.sum(b))
        return rms

    def _pack_kinematic_group(self):
        pn = self.parameter_names
        amplitudes = self._get_parameter_indices('amplitude')

        k_range = np.arange(1, self.parameters_per_feature).astype(int)
        kinematic_parameters = np.array([], dtype=int)
        kg_keys = list(self.kinematic_groups.keys())
        for g in kg_keys:
            ff = self.kinematic_groups[g][0]  # first feature name
            k_idx = pn.index((ff, 'amplitude')) + k_range
            kinematic_parameters = np.concatenate([kinematic_parameters, k_idx])
        transform = np.concatenate([amplitudes, kinematic_parameters])

        inverse_transform = np.array([])
        n_features = len(amplitudes)
        for i in range(len(self.feature_names)):
            feature_name = pn[int(i * self.parameters_per_feature)][0]
            for j, k in enumerate(kg_keys):
                if feature_name in self.kinematic_groups[k]:
                    k_idx = n_features + (j * (self.parameters_per_feature - 1)) + k_range - 1
                    inverse_transform = np.concatenate([inverse_transform, [i], k_idx])

        self.kinematic_group_transform = transform.astype(int)
        self.kinematic_group_inverse_transform = inverse_transform.astype(int)

    def _get_parameter_indices(self, regular_expression: str):
        return np.array([self.parameter_names.index(_) for _ in self.parameter_names if regular_expression in _])

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

        if self.kinematic_groups == {}:
            # TODO: Make initial guess reflect kinematic groups to output it correctly later.
            p_0 = self.initial_guess
            bounds = self.bounds
            self.kinematic_group_inverse_transform = Ellipsis
        else:
            if self.kinematic_group_transform is None:
                self._pack_kinematic_group()
            p_0 = self.initial_guess[self.kinematic_group_transform]
            bounds = [self.bounds[_] for _ in self.kinematic_group_transform]

        constraints = self._evaluate_constraints()

        s = self.data[~self.mask] - self.pseudo_continuum[~self.mask] - self.stellar[~self.mask]
        # noinspection PyTypeChecker
        solution = minimize(self.res, x0=p_0, args=(s,), method=min_method, bounds=bounds, constraints=constraints,
                            options=minimize_options)
        p = solution.x[self.kinematic_group_inverse_transform]
        self.solution = p

        if verbose:
            for i, j in enumerate(self.parameter_names):
                if j[1] == 'amplitude':
                    print(f'{j[0]}.{j[1]} = {p[i] * self.flux_scale_factor:8.2e}')
                else:
                    print(f'{j[0]}.{j[1]} = {p[i]:.2f}')

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

        p = self.solution
        u = self.uncertainties

        if verbose:
            for i, j in enumerate(self.parameter_names):
                if j[1] == 'amplitude':
                    print(
                        f'{j[0]}.{j[1]} = {p[i] * self.flux_scale_factor:8.2e}'
                        f'+- {u[i] * self.flux_scale_factor:8.2e}')
                else:
                    print(f'{j[0]}.{j[1]} = {p[i]:.2f} +- {u[i]:.2f}')

    def _get_solution_parameter(self, feature, parameter):
        return self.solution[self.parameter_names.index((feature, parameter))]

    def _get_feature_solution(self, feature):
        i = self.parameter_names.index((feature, 'amplitude'))
        return self.solution[i:i + self.parameters_per_feature]

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
            if np.any(np.isnan(self._get_feature_solution(component))):
                flux_model[idx] = np.nan
                flux_direct[idx] = np.nan

            else:
                z = (1.0 + self._get_solution_parameter(component, 'velocity') / c)
                center_wavelength = np.array([self.feature_wavelengths[idx] * z])
                sigma_vel = np.array([self._get_solution_parameter(component, 'sigma')])
                sigma_lam = spectools.sigma_lambda(sigma_vel, center_wavelength)
                mask = spectools.feature_mask(wavelength=self.wavelength, feature_wavelength=center_wavelength,
                                              sigma=sigma_lam, width=sigma_factor)

                observed_feature = self.data[~mask] - self.pseudo_continuum[~mask] - self.stellar[~mask]
                fit = self.function(self.wavelength[~mask], self.feature_wavelengths[idx],
                                    self._get_feature_solution(component))
                for i, j in enumerate(self.feature_names):
                    if j != component:
                        observed_feature -= self.function(
                            self.wavelength[~mask], self.feature_wavelengths[i], self._get_feature_solution(j))

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
