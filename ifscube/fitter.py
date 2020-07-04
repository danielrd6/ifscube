#!/usr/bin/env python
import argparse
import ctypes
import os

from . import Cube
from . import cubetools
from . import gmos
from . import manga
from . import modeling
from . import onedspec
from . import parser
from . import spectools


def make_lock(fname):
    with open(fname + '.lock', 'w') as f:
        f.write('This one is taken, go to the next.\n')
    return


def clear_lock(lockname):
    if os.path.isfile(lockname):
        os.remove(lockname)

    return


def spectrum_fit(data: onedspec.Spectrum, **line_fit_args):
    general_fit_args = {_: line_fit_args[_]for _ in ['function', 'fit_continuum', 'fitting_window']
                        if _ in line_fit_args.keys()}
    general_fit_args['continuum_options'] = line_fit_args['copts']
    fit = modeling.LineFit(data, **general_fit_args)

    for feature in line_fit_args['features']:
        fit.add_feature(**feature)

    for bounds in line_fit_args['bounds']:
        fit.set_bounds(*bounds)

    for constraint in line_fit_args['constraints']:
        fit.add_minimize_constraint(*constraint)

    if line_fit_args['optimize_fit']:
        fit.optimize_fit(width=line_fit_args['optimization_window'])

    if line_fit_args['fixed']:
        fit.solution = fit.initial_guess
        print('Not fitting! Returning initial guess.')
        fit.print_parameters('solution')
    else:
        fit.fit(min_method=line_fit_args['method'], minimize_options=line_fit_args['minopts'], verbose=True)

    if line_fit_args['monte_carlo']:
        fit.monte_carlo(line_fit_args['monte_carlo'], verbose=True)
    return fit


def cube_fit(data: Cube, **line_fit_args):
    assert 1 == 0, 'NOT IMPLEMENTED YET!'


def dofit(fname, linefit_args, overwrite, cubetype, loading, fit_type, config_file_name, plot=False, lock=False):
    galname = fname.split('/')[-1]

    try:
        suffix = linefit_args['suffix']
    except KeyError:
        suffix = None

    try:
        outname = linefit_args['out_image']
    except KeyError:
        outname = None

    if outname is None:
        if suffix is None:
            suffix = '_linefit'
        outname = galname.replace('.fits', suffix + '.fits')

    if not overwrite:
        if os.path.isfile(outname):
            print('ERROR! File {:s} already exists.'.format(outname))
            return

    lockname = outname + '.lock'
    if lock:
        if os.path.isfile(lockname):
            print('ERROR! Lock file {:s} is present.'.format(lockname))
            return
        else:
            make_lock(outname)

    if overwrite:
        if os.path.isfile(outname):
            os.remove(outname)
    else:
        if os.path.isfile(outname):
            if lock:
                clear_lock(lockname)
            return

    if fit_type == 'cube':

        if cubetype is None:
            a = Cube(fname, **loading)
        elif cubetype == 'manga':
            a = manga.cube(fname, **loading)
        elif cubetype == 'gmos':
            a = gmos.Cube(fname, **loading)
        else:
            raise RuntimeError('cubetype "{:s}" not understood.'.format(cubetype))
        fit_function = cube_fit

    elif fit_type == 'spec':

        if cubetype is None:
            a = onedspec.Spectrum(fname, **loading)
        elif cubetype == 'intmanga':
            a = manga.IntegratedSpectrum(fname, **loading)
        else:
            raise RuntimeError('cubetype "{:s}" not understood.'.format(cubetype))
        fit_function = spectrum_fit

    else:
        raise RuntimeError('fit_type "{:s}" not understood.'.format(fit_type))

    linefit_args['out_image'] = outname

    if 'weights' in linefit_args['copts']:
        linefit_args['copts']['weights'] = spectools.read_weights(a.rest_wavelength, linefit_args['copts']['weights'])

    fit = fit_function(data=a, **linefit_args)
    try:
        cubetools.append_config(config_file_name, outname)
    except IOError:
        if lock:
            clear_lock(lockname)

    if plot:
        fit.plot()

    if lock:
        clear_lock(lockname)

    return


def main(fit_type):
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--overwrite', action='store_true', help='Overwrites previous fit with the same name.')
    ap.add_argument('-p', '--plot', action='store_true', help='Plots the resulting fit.')
    ap.add_argument('-l', '--lock', action='store_true', default=False, help='Creates a lock file to prevent multiple '
                                                                             'instances fromattempting to fit the same file at the same time.')
    ap.add_argument('-b', '--cubetype', type=str, default=None, help='"gmos" or "manga".')
    ap.add_argument('-t', '--mklthreads', type=int, default=1, help='Number of threads for numpy routines.')
    ap.add_argument('-c', '--config', type=str, help='Config file.')
    ap.add_argument('datafile', help='FITS data file to be fit.', nargs='*')

    args = ap.parse_args()

    try:
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(args.mklthreads)))
    except OSError:
        print('WARNING!: Not setting the number of threads.')

    for i in args.datafile:
        c = parser.LineFitParser(args.config)
        line_fit_args = c.get_vars()
        dofit(i, line_fit_args, overwrite=args.overwrite, cubetype=args.cubetype, plot=args.plot,
              loading=c.loading_opts, lock=args.lock, fit_type=fit_type, config_file_name=args.config)
        del c
