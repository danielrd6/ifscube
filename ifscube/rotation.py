import configparser
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
                self.obs = self.obs[int(plane)]

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
        parameters: dict
            Dictionary of parameters.
        """

        for key in parameters:
            n = list(self.model.param_names).index(key)
            self.model.parameters[n] = parameters[key]

    def fit_model(self):
        """
        Fits a rotation model to the data.
        """
        fit = LevMarLSQFitter()
        self.solution = fit(self.model, self.x, self.y, self.obs)
        self.best_fit = self.solution(self.x, self.y)

    def plot_results(self):
        """Plots the fit results."""

        fig = plt.figure()

        for i, data in zip(range(1, 4), [self.obs, self.best_fit, self.obs - self.best_fit]):
            ax = fig.add_subplot(1, 3, i)
            ax.set_aspect('equal')
            ax.contourf(data, cmap='Spectral_r')

        plt.show()


def main():
    """
    Fits a rotation model to a 2D array of velocities.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='Configuration file.')

    args = ap.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    r = Rotation(**config['loading'])
    r.update_model(config['model'])
    r.fit_model()
    print(r.solution)
    r.plot_results()
