import argparse
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

import ifscube.io.line_fit
from . import Cube
from . import cubetools
from . import onedspec
from . import parser
from . import spectools


def make_lock(file_name):
    with open(file_name + '.lock', 'w') as f:
        f.write('This one is taken, go to the next.\n')
    return


def clear_lock(lock_name):
    if os.path.isfile(lock_name):
        os.remove(lock_name)

    return


def spectrum_fit(data: Union[Cube, onedspec.Spectrum], **line_fit_args):
    fit = ifscube.io.line_fit.setup_fit(data, **line_fit_args)

    assert fit.fitting_window[0] < np.max(fit.wavelength), \
        f"Lower limit of fitting window above maximum wavelength: {fit.fitting_window[0]} >= {np.max(fit.wavelength)}"

    assert fit.fitting_window[1] > np.min(fit.wavelength), \
        f"Upper limit of fitting window below minimum wavelength: {fit.fitting_window[1]} <= {np.min(fit.wavelength)}"

    if line_fit_args['fit_continuum']:
        continuum_options = line_fit_args['copts'] if line_fit_args['copts'] is not None else {}
        fit.fit_pseudo_continuum(**continuum_options)
    line_fit_args.pop('fit_continuum')
    if line_fit_args['fixed']:
        print('Not fitting! Returning initial guess.')
        fit.fit(min_method='fixed', verbose=line_fit_args['verbose'])
    else:
        if line_fit_args['monte_carlo']:
            print('\n' + (40 * '-') + '\n' + 'Initial fit.\n')
        fit.fit(min_method=line_fit_args['method'], minimum_good_fraction=line_fit_args["minimum_good_fraction"],
                verbose=line_fit_args['verbose'], **line_fit_args["minimization_options"])

    fit.integrate_flux()
    fit.equivalent_width()

    if line_fit_args['monte_carlo']:
        if line_fit_args['monte_carlo']:
            print('\n' + (40 * '-') + '\n' + f'Monte carlo run with {line_fit_args["monte_carlo"]} iterations.\n')
        fit.monte_carlo(line_fit_args['monte_carlo'])

    if line_fit_args['write_fits']:
        kwargs = {_: line_fit_args[_] for _ in ['out_image', 'suffix', 'function', 'overwrite']}
        ifscube.io.line_fit.write_spectrum_fit(fit, **kwargs)

    return fit


def do_fit(file_name, line_fit_args, overwrite, loading, fit_type, config_file_name, plot=False, lock=False,
           plot_all=True):
    data_file = os.path.basename(file_name)

    try:
        suffix = line_fit_args['suffix']
    except KeyError:
        suffix = None

    try:
        output_name = line_fit_args['out_image']
    except KeyError:
        output_name = None

    if output_name is None:
        if suffix is None:
            suffix = '_linefit'
        output_name = data_file.replace('.fits', suffix + '.fits')

    if not overwrite:
        if os.path.isfile(output_name):
            print('ERROR! File {:s} already exists.'.format(output_name))
            return

    lock_name = output_name + '.lock'
    if lock:
        if os.path.isfile(lock_name):
            print('ERROR! Lock file {:s} is present.'.format(lock_name))
            return
        else:
            make_lock(output_name)

    if overwrite:
        if os.path.isfile(output_name):
            os.remove(output_name)
    else:
        if os.path.isfile(output_name):
            if lock:
                clear_lock(lock_name)
            return

    if fit_type == 'cube':
        a = Cube(file_name, **loading)
    elif fit_type == 'spec':
        a = onedspec.Spectrum(file_name, **loading)
    else:
        raise RuntimeError('fit_type "{:s}" not understood.'.format(fit_type))

    line_fit_args['out_image'] = output_name

    if 'weights' in line_fit_args['copts']:
        line_fit_args['copts']['weights'] = spectools.read_weights(a.rest_wavelength, line_fit_args['copts']['weights'])

    fit = spectrum_fit(data=a, **line_fit_args)
    try:
        cubetools.append_config(config_file_name, output_name)
    except IOError:
        if lock:
            clear_lock(lock_name)

    if plot:
        fit.plot(plot_all=plot_all)
        plt.show()

    if lock:
        clear_lock(lock_name)

    return


def main(fit_type):
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str, help="Config file.")
    ap.add_argument("-f", "--focused-plot", action="store_false",
                    help="Plots just the fitting window, not the whole spectrum.")
    ap.add_argument("-l", "--lock", action="store_true", default=False,
                    help="Creates a lock file to prevent multiple instances from attempting to fit the same file at "
                         "the same time.")
    ap.add_argument("-o", "--overwrite", action="store_true", help="Overwrites previous fit with the same name.")
    ap.add_argument("-p", "--plot", action="store_true", help="Plots the resulting fit.")
    ap.add_argument("datafile", help="FITS data file to be fit.", nargs="*")

    args = ap.parse_args()

    for i in args.datafile:
        c = parser.LineFitParser(args.config)
        line_fit_args = c.get_vars()
        do_fit(i, line_fit_args, overwrite=args.overwrite, plot=args.plot, loading=c.loading_opts, lock=args.lock,
               fit_type=fit_type, config_file_name=args.config, plot_all=args.focused_plot)
