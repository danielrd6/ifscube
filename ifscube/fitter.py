import warnings
import argparse
import os

import ifscube.io.line_fit
from . import Cube
from . import cubetools
from . import gmos
from . import manga
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


def spectrum_fit(data: onedspec.Spectrum, **line_fit_args):
    fit = ifscube.io.line_fit.setup_fit(data, **line_fit_args)

    if line_fit_args['fixed']:
        fit.solution = fit.initial_guess
        print('Not fitting! Returning initial guess.')
        fit.print_parameters('solution')
    else:
        if line_fit_args['monte_carlo']:
            print('\n' + (40 * '-') + '\n' + 'Initial fit.\n')
        fit.fit(min_method=line_fit_args['method'], minimize_options=line_fit_args['minopts'], verbose=True)

    if line_fit_args['monte_carlo']:
        if line_fit_args['monte_carlo']:
            print('\n' + (40 * '-') + '\n' + f'Monte carlo run with {line_fit_args["monte_carlo"]} iterations.\n')
        fit.monte_carlo(line_fit_args['monte_carlo'], verbose=True)

    if line_fit_args['write_fits']:
        args = {_: line_fit_args[_] for _ in ['out_image', 'suffix', 'function', 'overwrite']}
        ifscube.io.line_fit.write_spectrum_fit(data, fit, args)

    return fit


def cube_fit(data: Cube, **line_fit_args):
    fit = ifscube.io.line_fit.setup_fit(data, **line_fit_args)

    if line_fit_args['fixed']:
        fit.solution = fit.initial_guess
        print('Not fitting! Returning initial guess.')
        fit.print_parameters('solution')
    else:
        if line_fit_args['monte_carlo']:
            print('\n' + (40 * '-') + '\n' + 'Initial fit.\n')
        fit.fit(min_method=line_fit_args['method'], minimize_options=line_fit_args['minopts'], verbose=False,
                fit_continuum=line_fit_args['fit_continuum'], continuum_options=line_fit_args['copts'])

    if line_fit_args['monte_carlo']:
        warnings.warn('Monte carlo not implemented yet!')
        # if line_fit_args['monte_carlo']:
        #     print('\n' + (40 * '-') + '\n' + f'Monte carlo run with {line_fit_args["monte_carlo"]} iterations.\n')
        # fit.monte_carlo(line_fit_args['monte_carlo'], verbose=True)

    if line_fit_args['write_fits']:
        args = {_: line_fit_args[_] for _ in ['out_image', 'suffix', 'function', 'overwrite']}
        ifscube.io.line_fit.write_spectrum_fit(data, fit, args)

    return fit


def dofit(file_name, line_fit_args, overwrite, cube_type, loading, fit_type, config_file_name, plot=False, lock=False):
    galname = file_name.split('/')[-1]

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
        output_name = galname.replace('.fits', suffix + '.fits')

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

        if cube_type is None:
            a = Cube(file_name, **loading)
        elif cube_type == 'manga':
            a = manga.cube(file_name, **loading)
        elif cube_type == 'gmos':
            a = gmos.Cube(file_name, **loading)
        else:
            raise RuntimeError('cubetype "{:s}" not understood.'.format(cube_type))
        fit_function = cube_fit

    elif fit_type == 'spec':

        if cube_type is None:
            a = onedspec.Spectrum(file_name, **loading)
        elif cube_type == 'intmanga':
            a = manga.IntegratedSpectrum(file_name, **loading)
        else:
            raise RuntimeError('cubetype "{:s}" not understood.'.format(cube_type))
        fit_function = spectrum_fit

    else:
        raise RuntimeError('fit_type "{:s}" not understood.'.format(fit_type))

    line_fit_args['out_image'] = output_name

    if 'weights' in line_fit_args['copts']:
        line_fit_args['copts']['weights'] = spectools.read_weights(a.rest_wavelength, line_fit_args['copts']['weights'])

    fit = fit_function(data=a, **line_fit_args)
    try:
        cubetools.append_config(config_file_name, output_name)
    except IOError:
        if lock:
            clear_lock(lock_name)

    if plot:
        fit.plot()

    if lock:
        clear_lock(lock_name)

    return


def main(fit_type):
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--overwrite', action='store_true', help='Overwrites previous fit with the same name.')
    ap.add_argument('-p', '--plot', action='store_true', help='Plots the resulting fit.')
    ap.add_argument('-l', '--lock', action='store_true', default=False,
                    help='Creates a lock file to prevent multiple instances from attempting to fit the same file at '
                         'the same time.')
    ap.add_argument('-b', '--cubetype', type=str, default=None, help='"gmos" or "manga".')
    ap.add_argument('-c', '--config', type=str, help='Config file.')
    ap.add_argument('datafile', help='FITS data file to be fit.', nargs='*')

    args = ap.parse_args()

    for i in args.datafile:
        c = parser.LineFitParser(args.config)
        line_fit_args = c.get_vars()
        dofit(i, line_fit_args, overwrite=args.overwrite, cube_type=args.cubetype, plot=args.plot,
              loading=c.loading_opts, lock=args.lock, fit_type=fit_type, config_file_name=args.config)
