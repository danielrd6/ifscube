import configparser
from typing import Union

import numpy as np
from astropy import table
from astropy.io import fits

from ifscube import onedspec, parser, modeling, datacube


def setup_fit(data: Union[onedspec.Spectrum, datacube.Cube], **line_fit_args):
    general_fit_args = {_: line_fit_args[_] for _ in ['function', 'fitting_window', 'instrument_dispersion']
                        if _ in line_fit_args.keys()}
    if isinstance(data, onedspec.Spectrum):
        fit = modeling.LineFit(data, **general_fit_args)
    elif isinstance(data, datacube.Cube):
        fit = modeling.LineFit3D(data, **general_fit_args)

    for feature in line_fit_args['features']:
        fit.add_feature(**feature)
    for bounds in line_fit_args['bounds']:
        fit.set_bounds(*bounds)
    for constraint in line_fit_args['constraints']:
        fit.add_minimize_constraint(*constraint)
    if line_fit_args['optimize_fit']:
        fit.optimize_fit(width=line_fit_args['optimization_window'])

    return fit


def write_spectrum_fit(spectrum, fit, args):
    suffix = args['suffix']
    out_image = args['out_image']
    if out_image is None:
        if suffix is None:
            suffix = '_linefit'
        out_image = spectrum.fitsfile.replace('.fits', suffix + '.fits')

    hdr = spectrum.header
    try:
        hdr['REDSHIFT'] = spectrum.redshift
    except KeyError:
        hdr['REDSHIFT'] = (spectrum.redshift,
                           'Redshift used in IFSCUBE')

    # Creates MEF output.
    h = fits.HDUList()
    hdu = fits.PrimaryHDU(header=spectrum.header)
    hdu.name = 'PRIMARY'
    h.append(hdu)

    # Creates the fitted spectrum extension
    hdr = fits.Header()
    hdr['object'] = ('spectrum', 'Data in this extension')
    hdr['CRPIX1'] = (1, 'Reference pixel for wavelength')
    hdr['CRVAL1'] = (fit.wavelength[0], 'Reference value for wavelength')

    d_wl = np.diff(fit.wavelength)
    avg_d_wl = np.average(d_wl)
    assert np.all(np.abs((d_wl / avg_d_wl) - 1) < 1e-6), 'Wavelength vector is not linearly sampled.'

    hdr['CD1_1'] = (avg_d_wl, 'CD1_1')
    hdu = fits.ImageHDU(data=fit.data, header=hdr)
    hdu.name = 'FITSPEC'
    h.append(hdu)

    hdu = fits.ImageHDU(data=fit.variance, header=hdr)
    hdr['object'] = ('variance', 'Data in this extension')
    hdu.name = 'VAR'
    h.append(hdu)

    # Creates the fitted continuum extension.
    hdr['object'] = 'continuum'
    hdu = fits.ImageHDU(data=fit.pseudo_continuum, header=hdr)
    hdu.name = 'FITCONT'
    h.append(hdu)

    # Creates the stellar continuum extension.
    hdr['object'] = 'stellar'
    hdu = fits.ImageHDU(data=fit.stellar, header=hdr)
    hdu.name = 'STELLAR'
    h.append(hdu)

    # Creates the fitted function extension.
    hdr['object'] = 'modeled_spec'
    hdu = fits.ImageHDU(data=fit.model_spectrum, header=hdr)
    hdu.name = 'MODEL'
    h.append(hdu)

    # Creates the solution extension.
    hdr = fits.Header()
    function = args['function']
    hdr['fitstat'] = fit.status
    total_pars = fit.solution.shape[0]

    hdr['object'] = 'parameters'
    hdr['function'] = (function, 'Fitted function')
    hdr['nfunc'] = (int(total_pars / fit.parameters_per_feature), 'Number of functions')
    hdu = fits.ImageHDU(data=fit.solution, header=hdr)
    hdu.name = 'SOLUTION'
    h.append(hdu)

    if fit.uncertainties is not None:
        hdr['object'] = 'dispersion'
        hdr['function'] = (function, 'Fitted function')
        hdr['nfunc'] = (int(total_pars / spectrum.npars), 'Number of functions')
        hdu = fits.ImageHDU(data=fit.uncertainties, header=hdr)
        hdu.name = 'DISP'
        h.append(hdu)

    # Integrated flux extensions
    hdr['object'] = 'flux_model'
    hdu = fits.ImageHDU(data=fit.flux_model, header=hdr)
    hdu.name = 'FLUX_M'
    h.append(hdu)

    hdr['object'] = 'flux_direct'
    hdu = fits.ImageHDU(data=fit.flux_direct, header=hdr)
    hdu.name = 'FLUX_D'
    h.append(hdu)

    # Equivalent width extensions
    hdr['object'] = 'eqw_model'
    hdu = fits.ImageHDU(data=fit.eqw_model, header=hdr)
    hdu.name = 'EQW_M'
    h.append(hdu)

    hdr['object'] = 'eqw_direct'
    hdu = fits.ImageHDU(data=fit.eqw_direct, header=hdr)
    hdu.name = 'EQW_D'
    h.append(hdu)

    # Creates component and parameter names table.
    fit_table = table.Table(np.array(fit.parameter_names), names=('component', 'parameter'))
    hdr['object'] = 'parameter names'
    hdu = fits.table_to_hdu(fit_table)
    hdu.name = 'PARNAMES'
    h.append(hdu)

    h.writeto(out_image, overwrite=args['overwrite'], checksum=True)


def table_to_config(t: table.Table):
    cfg = configparser.ConfigParser()
    previous_section = None
    for line in t:
        section, option = line['parameters'].split('.')
        value = line['values']
        if section != previous_section:
            cfg.add_section(section)
            previous_section = section
        sec = cfg[section]
        sec[option] = value
    return cfg


def load_fit(file_name):
    """
    Loads the result of a previous fit, and put it in the
    appropriate variables for the plotfit function.

    Parameters
    ----------
    file_name : string
        Name of the FITS file containing the fit results.

    Returns
    -------
    Nothing.
    """
    spectrum = onedspec.Spectrum(file_name, scidata='FITSPEC', variance='VAR', stellar='STELLAR')
    with fits.open(file_name, mode='readonly') as h:
        config = table_to_config(table.Table(h['FITCONFIG'].data))
        fit = setup_fit(spectrum, **parser.LineFitParser(config).get_vars())
        fit.status = h['SOLUTION'].header['fitstat']
        translate_extensions = {'pseudo_continuum': 'fitcont', 'model': 'model', 'solution': 'solution',
                                'eqw_model': 'eqw_m', 'eqw_direct': 'eqw_d', 'flux_model': 'flux_m',
                                'flux_direct': 'flux_d'}
        for key in translate_extensions:
            setattr(fit, key, h[translate_extensions[key]].data)
    return fit
