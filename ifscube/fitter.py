#!/usr/bin/env python
# STDLIB
import os
import argparse
import ctypes
import configparser

# third party
from astropy.io import fits
from astropy import table

# LOCAL
from . import manga
from . import gmos
from . import parser
from . import onedspec
from . import Cube


def mklock(fname):
    with open(fname + '.lock', 'w') as f:
        f.write('This one is taken, go to the next.\n')
    return


def append_config(config_file, outname):

    c = configparser.ConfigParser()
    c.read(config_file)
    t = table.Table(names=['parameters', 'values'], dtype=('S64', 'S64'))

    for i in c.sections():
        for j in c[i]:
            t.add_row(('{:s}.{:s}'.format(i, j), c[i][j]))

    with fits.open(outname) as outfits:
        outfits.append(fits.BinTableHDU(data=t))
        outfits[-1].name = 'FITCONFIG'
        outfits.writeto(outname, overwrite=True)

    return


def dofit(fname, linefit_args, overwrite, cubetype, loading,
          fit_type, config_file_name, plot=False, lock=False):

    galname = fname.split('/')[-1]

    try:
        suffix = linefit_args['suffix']
    except KeyError:
        suffix = None

    try:
        outname = linefit_args['outimage']
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

    if lock:
        lockname = outname + '.lock'
        if os.path.isfile(lockname):
            print('ERROR! Lock file {:s} is present.'.format(lockname))
            return
        else:
            mklock(outname)

    if overwrite:
        if os.path.isfile(outname):
            os.remove(outname)
    else:
        if os.path.isfile(outname):
            if lock:
                if os.path.isfile(lockname):
                    os.remove(lockname)
            return

    if fit_type == 'cube':

        if cubetype is None:
            a = Cube(fname, **loading)
        elif cubetype == 'manga':
            a = manga.cube(fname, **loading)
        elif cubetype == 'gmos':
            a = gmos.cube(fname, **loading)

    elif fit_type == 'spec':

        if cubetype is None:
            a = onedspec.Spectrum(fname, **loading)
        elif cubetype == 'intmanga':
            a = manga.IntegratedSpectrum(fname, **loading)

    linefit_args['outimage'] = outname
    try:
        a.linefit(**linefit_args)
    except IOError:
        if lock:
            if os.path.isfile(lockname):
                os.remove(lockname)
        return

    append_config(config_file_name, outname)

    if plot:
        a.plotfit()

    if lock:
        if os.path.isfile(lockname):
            os.remove(lockname)

    return


def main(fit_type):

    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-o', '--overwrite', action='store_true',
        help='Overwrites previous fit with the same name.')
    ap.add_argument(
        '-p', '--plot', action='store_true',
        help='Plots the resulting fit.')
    ap.add_argument(
        '-l', '--lock', action='store_true', default=False,
        help='Creates a lock file to prevent multiple instances from'
        ' attempting to fit the same file at the same time.')
    ap.add_argument(
        '-b', '--cubetype', type=str, default=None,
        help='"gmos" or "manga".')
    ap.add_argument(
        '-t', '--mklthreads', type=int, default=1,
        help='Number of threads for numpy routines.')
    ap.add_argument('-c', '--config', type=str, help='Config file.')
    ap.add_argument('datafile', help='FITS data file to be fit.', nargs='*')

    args = ap.parse_args()

    c = parser.LineFitParser(args.config)

    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_rt.mkl_set_num_threads(
        ctypes.byref(
            ctypes.c_int(args.mklthreads)))

    for i in args.datafile:
        linefit_args = c.get_vars()
        dofit(
            i, linefit_args, overwrite=args.overwrite, cubetype=args.cubetype,
            plot=args.plot, loading=c.loading_opts, lock=args.lock,
            fit_type=fit_type, config_file_name=args.config)
