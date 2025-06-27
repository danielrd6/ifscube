import argparse
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib.widgets import Slider, TextBox, Button
from numpy import ma
import json

from ifscube.models import DiskRotation


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
                    self.obs = ma.masked_invalid(h[extension].data)
            elif isinstance(input_data, np.ndarray):
                self.obs = ma.masked_invalid(input_data)
            elif isinstance(input_data, ma.masked_array):
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
            self.obs = ma.array([])

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
            setattr(self.model, key, parameters[key])

    def update_bounds(self, bounds):
        for key in bounds:
            self.model.bounds[key] = bounds[key]

    def update_fixed(self, fixed):
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
            value = self.solution.parameters[n]
            if key in ['phi_0', 'theta']:
                value = np.rad2deg(value)
            color = '\033[91m' if self.solution.parameters[n] in self.model.bounds[key] else ''
            print(f'{color}{key:12s}: {value:8.3f}\033[0m')

    def plot_results(self, contrast=1.0, contours=True, symmetric=True):
        """Plots the fit results."""

        vmin = np.percentile(self.obs[~self.obs.mask], contrast)
        vmax = np.percentile(self.obs[~self.obs.mask], 100.0 - contrast)

        if symmetric:
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

        fig.colorbar(im, ax=axes[1], orientation='horizontal')
        plt.show()

    def interactive_guess(self, contrast=1.0):
        v_min = np.percentile(self.obs[~self.obs.mask], contrast)
        v_max = np.percentile(self.obs[~self.obs.mask], 100.0 - contrast)
        kwargs = dict(cmap='Spectral_r', vmin=v_min, vmax=v_max)

        m = np.max(np.abs([v_min, v_max]))
        v_min = -m
        v_max = m

        velocity_map = self.obs

        # -- Initial parameter values --
        params_init = {
            'amplitude': 200.0,
            'x_0': 0.0,
            'y_0': 0.0,
            'phi_0': 0.0,
            'c_0': 1.0,
            'p': 1.2,
            'theta': 45.0,
        }

        # -- Parameter ranges for sliders --
        param_ranges = {
            'amplitude': (100, 500),
            'x_0': (-300, 300),
            'y_0': (-300, 300),
            'phi_0': (0, 180),
            'c_0': (1.0, 10.0),
            'p': (1.0, 1.5),
            'theta': (0.0, 90.0),
        }

        # -- Create figure layout --
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(4, 3)

        ax_obs = fig.add_subplot(gs[0:2, 0])
        ax_model = fig.add_subplot(gs[0:2, 1])
        ax_resid = fig.add_subplot(gs[0:2, 2])

        # -- Display observed image --
        im_obs = ax_obs.imshow(velocity_map, origin='lower', **kwargs)
        ax_obs.set_title("Observed")

        # -- Call external model --
        def get_model(params):
            for key in params:
                if key in ['phi_0', 'theta']:
                    params[key] = np.deg2rad(params[key])

            return DiskRotation(**params)(self.x, self.y)

        model = get_model(params_init)
        im_model = ax_model.imshow(model, origin='lower', **kwargs)
        ax_model.set_title("Model")

        residuals = velocity_map - model
        im_resid = ax_resid.imshow(residuals, origin='lower', **kwargs)
        ax_resid.set_title("Residuals")

        # Sliders + TextBoxes
        sliders = {}
        textboxes = {}

        slider_height = 0.025
        spacing = 0.035
        bottom_start = 0.28

        for i, (param, val) in enumerate(params_init.items()):
            y_pos = bottom_start - i * spacing

            # Slider (left 70% of the width)
            ax_slider = plt.axes([0.15, y_pos, 0.6, slider_height])
            slider = Slider(ax_slider, param, *param_ranges[param], valinit=val)
            sliders[param] = slider

            # TextBox (right 10%)
            ax_box = plt.axes([0.85, y_pos, 0.08, slider_height])
            textbox = TextBox(ax_box, '', initial=str(val))
            textboxes[param] = textbox

            # Link: slider → textbox
            def make_slider_callback(p):
                return lambda val: textboxes[p].set_val(f"{val:.3f}")

            slider.on_changed(make_slider_callback(param))

            # Link: textbox → slider
            def make_textbox_submit(p):
                def submit(text):
                    try:
                        num = float(text)
                        min_val, max_val = param_ranges[p]
                        if min_val <= num <= max_val:
                            sliders[p].set_val(num)
                        else:
                            print(f"{p}: Value out of range")
                    except ValueError:
                        print(f"{p}: Invalid input")

                return submit

            textbox.on_submit(make_textbox_submit(param))
        # -- Update on slider change --
        def update(val):
            current = {k: s.val for k, s in sliders.items()}
            new_model = get_model(current)
            im_model.set_data(new_model)
            im_resid.set_data(velocity_map - new_model)
            fig.canvas.draw_idle()

        for slider in sliders.values():
            slider.on_changed(update)

        # Save button
        ax_button = plt.axes([0.40, 0.01, 0.2, 0.04])
        save_button = Button(ax_button, 'Save')

        def save_params(event):
            current = {k: s.val for k, s in sliders.items()}
            with open('parameters.json', 'w') as f:
                json.dump(current, f, indent=2)
            print("Parameters saved to parameters.json")

        save_button.on_clicked(save_params)

        plt.tight_layout()
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
    ap.add_argument("-i", "--interactive", action='store_true',
                    help="Interactive mode for setting initial guess.")

    args = ap.parse_args()

    config = Config(args.config)

    r = Rotation(**config.loading)
    r.update_model(config.model)

    if args.interactive:
        r.interactive_guess()

    # if config.getboolean('general', 'fit'):
    #     r.update_bounds(config.bounds)
    #     r.update_fixed(config.fixed)
    #     r.fit_model(maxiter=1000)
    # else:
    #     r.solution = r.model
    #     r.best_fit = r.model(r.x, r.y)

    # r.print_solution()
    # r.plot_results(contours=False)

if __name__ == '__main__':
    main()