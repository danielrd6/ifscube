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
                    h_3: float = None, h_4: float = None, kinematic_group: int = None):

        if (rest_wavelength < self.fitting_window[0]) or (rest_wavelength > self.fitting_window[1]):
            warnings.warn(f'Spectral feature {name} outside of fitting window. Skipping.')
        else:
            self.feature_names.append(name)
            self.feature_wavelengths = np.concatenate([self.feature_wavelengths, [rest_wavelength]])

            if self.parameters_per_feature == 3:
                self.initial_guess = np.concatenate(
                    [self.initial_guess, [amplitude / self.flux_scale_factor, velocity, sigma]])
            elif self.parameters_per_feature == 5:
                self.initial_guess = np.concatenate(
                    [self.initial_guess, [amplitude / self.flux_scale_factor, velocity, sigma, h_3, h_4]])
                self.parameter_names += [f'{name}.{_}' for _ in ['amplitude', 'velocity', 'sigma', 'h_3', 'h_4']]

            self.parameter_names += [
                f'{name}.{_}' for _ in ['amplitude', 'velocity', 'sigma', 'h_3', 'h_4'][:self.parameters_per_feature]]
            self.bounds += [[None, None]] * self.parameters_per_feature

    def set_bounds(self, feature: str, parameter: str, bounds: list):
        if parameter == 'amplitude':
            bounds = [_ / self.flux_scale_factor if _ is not None else _ for _ in bounds]
        self.bounds[self.parameter_names.index(f'{feature}.{parameter}')] = bounds

    def add_minimize_constraint(self, parameter, constraint):
        self.constraint_expressions.append([parameter, constraint])

    def _evaluate_constraints(self):
        component_names = []
        parameter_names = []
        for i in self.parameter_names:
            c = i.split('.')
            if c[0] not in component_names:
                component_names.append(c[0])
            if c[1] not in parameter_names:
                parameter_names.append(c[1])

        constraints = []
        for c in self.constraint_expressions:
            expression = ' '.join([_ if '.' not in _ else _.split('.')[0] for _ in c[1].split()])
            cp = parser.ConstraintParser(expression, feature_names=component_names,
                                         parameter_names=parameter_names)
            feature, parameter = c[0].split('.')
            cp.evaluate(feature, parameter)
            constraints.append(cp.constraint)

        return constraints

    def res(self, x):
        m = self.function(self.wavelength[~self.mask], self.feature_wavelengths, x)
        s = self.data[~self.mask] - self.pseudo_continuum[~self.mask] - self.stellar[~self.mask]
        a = np.square((s - m) * self.weights[~self.mask])
        b = a / self.variance[~self.mask]
        rms = np.sqrt(np.sum(b))
        return rms

    def fit(self, min_method: str = 'slsqp', minimize_options: dict = None, verbose: bool = True):
        assert self.initial_guess.size > 0, 'There are no spectral features to fit. Aborting'
        assert self.initial_guess.size == (len(self.feature_names) * self.parameters_per_feature), \
            'There is a problem with the initial guess. Check the spectral feature definitions.'

        if minimize_options is None:
            minimize_options = {'eps': 1.0e-3}

        constraints = self._evaluate_constraints()
        self.solution = minimize(self.res, x0=self.initial_guess, method=min_method, bounds=self.bounds,
                                 constraints=constraints, options=minimize_options)

        if verbose:
            for i, j in enumerate(self.parameter_names):
                if 'amplitude' in j:
                    print(f'{j:<32s} = {self.solution.x[i] * self.flux_scale_factor:8.2e}')
                else:
                    print(f'{j:<32s} = {self.solution.x[i]:8.2f}')

    def plot(self):
        model = (self.function(self.wavelength, self.feature_wavelengths, self.solution.x)
                 + self.stellar + self.pseudo_continuum)[~self.mask] * self.flux_scale_factor

        fig = plt.figure()
        ax = fig.add_subplot(111)

        wavelength = self.wavelength[~self.mask]
        observed = (self.data * self.flux_scale_factor)[~self.mask]
        ax.plot(wavelength, observed)
        ax.plot(wavelength, model)

        plt.show()
