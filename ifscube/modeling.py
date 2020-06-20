import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from ifscube import elprofile, parser
from ifscube.spectools import continuum
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
            self.pseudo_continuum = continuum(self.wavelength, (self.data - self.stellar), output='function')[1]
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

    def optimization_mask(self):
        pass

    def add_feature(self, name: str, rest_wavelength: float, amplitude: float, velocity: float, sigma: float,
                    h_3: float = 0.0, h_4: float = 0.0, kinematic_group: int = None):

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

            if self.parameters_per_feature == 3:
                self.initial_guess = np.concatenate(
                    [self.initial_guess, [amplitude / self.flux_scale_factor, velocity, sigma]])
            elif self.parameters_per_feature == 5:
                self.initial_guess = np.concatenate(
                    [self.initial_guess, [amplitude / self.flux_scale_factor, velocity, sigma, h_3, h_4]])

            self.parameter_names += [
                f'{name}.{_}' for _ in ['amplitude', 'velocity', 'sigma', 'h_3', 'h_4'][:self.parameters_per_feature]]
            self.bounds += [[None, None]] * self.parameters_per_feature

    def set_bounds(self, feature: str, parameter: str, bounds: list):
        if parameter == 'amplitude':
            bounds = [_ / self.flux_scale_factor if _ is not None else _ for _ in bounds]
        self.bounds[self.parameter_names.index(f'{feature}.{parameter}')] = bounds

    def add_minimize_constraint(self, parameter, expression):
        self.constraint_expressions.append([parameter, expression])

    def _evaluate_constraints(self):
        if self.kinematic_groups != {}:
            pn = [self.parameter_names[_] for _ in self.kinematic_group_transform]
        else:
            pn = self.parameter_names

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
        amplitudes = np.array([pn.index(_) for _ in pn if 'amplitude' in _])

        k_range = np.arange(1, self.parameters_per_feature).astype(int)
        kinematic_parameters = np.array([], dtype=int)
        kg_keys = list(self.kinematic_groups.keys())
        for g in kg_keys:
            ff = self.kinematic_groups[g][0]  # first feature name
            k_idx = pn.index(f'{ff}.amplitude') + k_range
            kinematic_parameters = np.concatenate([kinematic_parameters, k_idx])
        transform = np.concatenate([amplitudes, kinematic_parameters])

        inverse_transform = np.array([])
        n_features = len(amplitudes)
        for i in range(int(len(pn) / self.parameters_per_feature)):
            feature_name = pn[int(i * self.parameters_per_feature)].split('.')[0]
            for j, k in enumerate(kg_keys):
                if feature_name in self.kinematic_groups[k]:
                    k_idx = n_features + (j * (self.parameters_per_feature - 1)) + k_range - 1
                    inverse_transform = np.concatenate([inverse_transform, [i], k_idx])

        self.kinematic_group_transform = transform.astype(int)
        self.kinematic_group_inverse_transform = inverse_transform.astype(int)

    def fit(self, min_method: str = 'slsqp', minimize_options: dict = None, verbose: bool = True):
        assert self.initial_guess.size > 0, 'There are no spectral features to fit. Aborting'
        assert self.initial_guess.size == (len(self.feature_names) * self.parameters_per_feature), \
            'There is a problem with the initial guess. Check the spectral feature definitions.'

        if minimize_options is None:
            minimize_options = {'eps': 1.0e-3}

        s = self.data[~self.mask] - self.pseudo_continuum[~self.mask] - self.stellar[~self.mask]

        if self.kinematic_groups != {}:
            self._pack_kinematic_group()
            p_0 = self.initial_guess[self.kinematic_group_transform]
            bounds = [self.bounds[_] for _ in self.kinematic_group_transform]
        else:
            p_0 = self.initial_guess
            bounds = self.bounds

        constraints = self._evaluate_constraints()
        # noinspection PyTypeChecker
        self.solution = minimize(self.res, x0=p_0, args=(s,), method=min_method, bounds=bounds,
                                 constraints=constraints, options=minimize_options)

        self.solution.x = self.solution.x[self.kinematic_group_inverse_transform]

        if verbose:
            for i, j in enumerate(self.parameter_names):
                if 'amplitude' in j:
                    print(f'{j:<32s} = {self.solution.x[i] * self.flux_scale_factor:8.2e}')
                else:
                    print(f'{j:<32s} = {self.solution.x[i]:8.2f}')

    def plot(self):

        wavelength = self.wavelength[~self.mask]
        observed = (self.data * self.flux_scale_factor)[~self.mask]
        pseudo_continuum = self.pseudo_continuum[~self.mask] * self.flux_scale_factor
        stellar = self.stellar[~self.mask] * self.flux_scale_factor
        model = self.function(wavelength, self.feature_wavelengths, self.solution.x) * self.flux_scale_factor

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(wavelength, observed)
        ax.plot(wavelength, model + stellar + pseudo_continuum)
        ax.plot(wavelength, pseudo_continuum)
        ax.plot(wavelength, stellar)

        ppf = self.parameters_per_feature
        for i in range(0, len(self.parameter_names), ppf):
            feature_wl = self.feature_wavelengths[int(i / ppf)]
            parameters = self.solution.x[i:i + ppf]
            line = self.function(wavelength, feature_wl, parameters) * self.flux_scale_factor
            ax.plot(wavelength, pseudo_continuum + stellar + line, 'k--')

        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Spectral flux density')
        ax.minorticks_on()

        ax.grid()
        plt.show()
