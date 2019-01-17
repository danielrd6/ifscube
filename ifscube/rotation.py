from configparser import ConfigParser
import argparse

from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
import matplotlib.pyplot as plt
import numpy as np

from .models import DiskRotation


class Rotation(object):
    """Class for fitting rotation models to velocity fields."""

    def __init__(self, input_data=None, extension=0, plane=0, model=None):
        """
        Initializes the instance and optionally loads the observation data.

        Parameters
        ----------
        input_data: str or numpy.ndarray
            Name of the FITS file containing the velocity field data or
            the data in the form of a numpy.ndarray.
        extension: integer or str
            Name or number of the extension of the FITS file which
            contains the velocity field.
        plane: int
            If the FITS file extension contains a data cube, specify the
            plane containing the correct velocity field with this parameter.
        model: astropy.modeling.Fittable2DModel
            Model to be fit. If *None* defaults to ifscube.models.DiskRotation.
        """

        if input_data is not None:

            if isinstance(input_data, str):
                with fits.open(input_data) as h:
                    self.obs = h[extension].data
            elif isinstance(input_data, np.ndarray):
                self.obs = input_data
            else:
                raise IOError('Could not understand input data.')

            dimensions = len(self.obs.shape)
            if dimensions not in [2, 3]:
                raise RuntimeError('Invalid format for input data. Dimensions must be 2 or 3.')
            if dimensions == 3:
                self.obs = self.obs[plane]

            self.y, self.x = np.indices(self.obs.shape)

        else:
            self.obs = np.array([])

        if model is None:
            self.model = DiskRotation()
        else:
            self.model = model

        self.solution = None
        self.best_fit = np.array([])

    def update_model(self, parameters):
        """
        Updates the model parameters.

        Parameters
        ----------
        parameters: configparser.Configparser
            Dictionary of parameters.
        """

        for key in parameters:
            n = list(self.model.param_names).index(key)
            self.model.parameters[n] = parameters[key]

    def update_bounds(self, bounds):
        for key in bounds:
            self.model.bounds[key] = bounds[key]

    def updated_fixed(self, fixed):
        for key in fixed:
            self.model.fixed[key] = fixed[key]

    def fit_model(self, maxiter=100):
        """
        Fits a rotation model to the data.
        """
        fit = LevMarLSQFitter()
        self.solution = fit(self.model, self.x, self.y, self.obs, maxiter=maxiter)
        self.best_fit = self.solution(self.x, self.y)

    def print_solution(self):
        for key in self.solution.param_names:
            n = self.solution.param_names.index(key)
            if key in ['phi_0', 'theta']:
                print('{:12s}: {:8.3f}'.format(key, np.rad2deg(self.solution.parameters[n])))
            else:
                print('{:12s}: {:8.3f}'.format(key, self.solution.parameters[n]))

    def plot_results(self, contrast=1.0, contours=True, symetric=True):
        """Plots the fit results."""

        vmin = np.percentile(self.obs, contrast)
        vmax = np.percentile(self.obs, 100.0 - contrast)

        if symetric:
            m = np.max(np.abs([vmin, vmax]))
            vmin = -m
            vmax = m

        fig, axes = plt.subplots(nrows=1, ncols=3, sharey='row')
        data = [self.obs, self.best_fit, self.obs - self.best_fit]
        labels = ['Observation', 'Model', 'Residual']

        kwargs = dict(cmap='Spectral_r', vmin=vmin, vmax=vmax)

        im = None
        for ax, d, lab in zip(axes, data, labels):
            ax.set_aspect('equal')
            if contours:
                im = ax.contourf(d, **kwargs)
            else:
                im = ax.imshow(d, origin='lower', **kwargs)
            ax.set_title(lab)

        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.05])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

        plt.show()


class Config(ConfigParser):

    def __init__(self, file_name):
        ConfigParser.__init__(self)
        self.read(file_name)

        if 'general' in self.sections():
            self._read_general()
        else:
            self.general = None

        if 'loading' in self.sections():
            self._read_loading()
        else:
            self.loading = None

        if 'model' in self.sections():
            self._read_model()
        else:
            self.general = None

        if 'bounds' in self.sections():
            self._read_bounds()
        else:
            self.bounds = None

        if 'fixed' in self.sections():
            self._read_fixed()
        else:
            self.fixed = None

    def _read_general(self):
        self.general = {'fit': self.getboolean('general', 'fit')}

    def _read_loading(self):
        self.loading = {'input_data': self.get('loading', 'input_data')}
        if 'extension' in self['loading']:
            e = self.get('loading', 'extension')
            self.loading['extension'] = (int(e) if e.isdigit() else e)
        self.loading['plane'] = self.getint('loading', 'plane')

    def _read_model(self):
        self.model = {key: self.getfloat('model', key) for key in self['model']}
        for key in ['phi_0', 'theta']:
            if key in self.model:
                self.model[key] = np.deg2rad(self.model[key])

    def _read_bounds(self):
        self.bounds = {}
        for key in self['bounds']:
            b = tuple([float(i) for i in self['bounds'][key].split(',')])
            if key in ['phi_0', 'theta']:
                b = tuple([np.deg2rad(i) for i in b])
            self.bounds.update({key: b})

    def _read_fixed(self):
        self.fixed = {}
        for key in self['fixed']:
            self.fixed.update({key: self.getboolean('fixed', key)})


def main():
    """
    Fits a rotation model to a 2D array of velocities.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='Configuration file.')

    args = ap.parse_args()

    config = Config(args.config)

    r = Rotation(**config.loading)
    r.update_model(config.model)

    if config.getboolean('general', 'fit'):
        r.update_bounds(config.bounds)
        r.updated_fixed(config.fixed)
        r.fit_model(maxiter=1000)
    else:
        r.solution = r.model
        r.best_fit = r.model(r.x, r.y)

    r.print_solution()
    r.plot_results(contours=False)
