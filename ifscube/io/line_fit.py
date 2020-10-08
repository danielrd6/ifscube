import warnings
import configparser
from typing import Union

import numpy as np
from astropy import table
from astropy.io import fits

from ifscube import onedspec, modeling, datacube, parser


def setup_fit(data: Union[onedspec.Spectrum, datacube.Cube], **line_fit_args):

    if isinstance(data, datacube.Cube):
        general_fit_args = {_: line_fit_args[_] for _ in ['function', 'fitting_window', 'instrument_dispersion',
                                                          'individual_spec', 'spiral_loop', 'spiral_center', 'refit',
                                                          'refit_radius', 'bounds_change']
                            if _ in line_fit_args.keys()}
        fit = modeling.LineFit3D(data, **general_fit_args)
    elif isinstance(data, onedspec.Spectrum):
        general_fit_args = {_: line_fit_args[_] for _ in ['function', 'fitting_window', 'instrument_dispersion']
                            if _ in line_fit_args.keys()}
        fit = modeling.LineFit(data, **general_fit_args)
    else:
        raise RuntimeError(f'Data instance "{str(data)}" not recognized.')

    for feature in line_fit_args['features']:
        fit.add_feature(**feature)
    for bounds in line_fit_args['bounds']:
        fit.set_bounds(*bounds)
    for constraint in line_fit_args['constraints']:
        fit.add_constraint(*constraint)
    if line_fit_args['optimize_fit']:
        fit.optimize_fit(width=line_fit_args['optimization_window'])

    return fit


def check_dimensions(data: np.ndarray):
    """
    Checks if data represents a 3D data cube.

    Parameters
    ----------
    data : np.ndarray
        Input data.

    Returns
    -------
    is_cube : bool
        True if data is 3D.
    """
    n_dimensions = len(data.data.shape)
    is_cube = n_dimensions == 3
    if not is_cube:
        assert len(data.data.shape) == 1, f'Data dimensions are expected to be either 1 or 3, got "{n_dimensions}.'
    return is_cube


def write_spectrum_fit(fit: Union[modeling.LineFit, modeling.LineFit3D], suffix: str = None, out_image: str = None,
                       function: str = None, overwrite: bool = False):
    """
    Writes the results in a LineFit instance to a FITS file.

    Parameters
    ----------
    fit : Union[modeling.LineFit, modeling.LineFit3D]
        LineFit instance.
    suffix : str
        Suffix to append to the end of the input FITS file if
        *out_image* is not given.
    out_image : str
        Name of the output FITS file. If *out_image* is given,
        then suffix is ignored.
    function : str
        Name of the function used to describe the spectral feature
        profiles.
    overwrite : bool
        Overwrites file with the same name.

    Returns
    -------
    None

    """
    is_cube = check_dimensions(fit.input_data.data)
    if out_image is None:
        if suffix is None:
            suffix = '_linefit'
        out_image = fit.input_data.fitsfile.replace('.fits', suffix + '.fits')

    hdr = fit.input_data.header
    try:
        hdr['REDSHIFT'] = fit.input_data.redshift
    except KeyError:
        hdr['REDSHIFT'] = (fit.input_data.redshift, 'Redshift used in IFSCUBE')

    # Creates MEF output.
    h = fits.HDUList()
    hdu = fits.PrimaryHDU(header=fit.input_data.header)
    hdu.name = 'PRIMARY'
    h.append(hdu)

    # Creates the fitted spectrum extension
    hdr = fits.Header()
    hdr['object'] = ('spectrum', 'Data in this extension')
    d_wl = np.diff(fit.wavelength)
    avg_d_wl = np.average(d_wl)
    assert np.all(np.abs((d_wl / avg_d_wl) - 1) < 1e-6), 'Wavelength vector is not linearly sampled.'
    if is_cube:
        hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
        hdr['CRVAL3'] = (fit.wavelength[0], 'Reference value for wavelength')
        hdr['CD3_3'] = (np.average(np.diff(fit.wavelength)), 'CD3_3')
    else:
        hdr['CRPIX1'] = (1, 'Reference pixel for wavelength')
        hdr['CRVAL1'] = (fit.wavelength[0], 'Reference value for wavelength')
        hdr['CD1_1'] = (avg_d_wl, 'CD1_1')
        fit.reduced_chi_squared = np.array([[fit.reduced_chi_squared]])
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
    if is_cube:
        hdr['fit_x0'] = fit.x_0
        hdr['fit_y0'] = fit.y_0
    total_pars = fit.solution.shape[0]

    hdr['object'] = 'parameters'
    hdr['function'] = (function, 'Fitted function')
    hdr['nfunc'] = (int(total_pars / fit.parameters_per_feature), 'Number of functions')
    hdu = fits.ImageHDU(data=fit.solution, header=hdr)
    hdu.name = 'SOLUTION'
    h.append(hdu)

    # Creates the reduced chi squared.
    hdr['object'] = 'Reduced Chi^2'
    hdu = fits.ImageHDU(data=fit.reduced_chi_squared, header=hdr)
    hdu.name = 'RED_CHI'
    h.append(hdu)

    if fit.uncertainties is not None:
        hdr['object'] = 'dispersion'
        hdu = fits.ImageHDU(data=fit.uncertainties, header=hdr)
        hdu.name = 'DISP'
        h.append(hdu)

    # Creates the initial guess extension.
    hdr['object'] = 'Initial guess'
    hdu = fits.ImageHDU(data=fit.initial_guess, header=hdr)
    hdu.name = 'INIGUESS'
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

    # Creates the minimize's exit status extension
    hdr['object'] = 'status'
    if isinstance(fit.status, int):
        # noinspection PyTypeChecker
        hdu = fits.ImageHDU(data=np.array([fit.status]), header=hdr)
    else:
        hdu = fits.ImageHDU(data=fit.status, header=hdr)
    hdu.name = 'STATUS'
    h.append(hdu)

    if is_cube:
        # Creates the spatial mask extension
        hdr['object'] = 'spatial mask'
        hdu = fits.ImageHDU(data=fit.input_data.spatial_mask.astype(int), header=hdr)
        hdu.name = 'MASK2D'
        h.append(hdu)

        # Creates the spaxel indices extension as fits.BinTableHDU.
        hdr['object'] = 'spaxel_coords'
        t = table.Table(fit.spaxel_indices, names=('row', 'column'))
        hdu = fits.table_to_hdu(t)
        hdu.name = 'SPECIDX'
        h.append(hdu)

    # Creates component and parameter names table.
    fit_table = table.Table(np.array(fit.parameter_names), names=('component', 'parameter'))
    hdr['object'] = 'parameter names'
    hdu = fits.table_to_hdu(fit_table)
    hdu.name = 'PARNAMES'
    h.append(hdu)

    fit_table = table.Table([table.Column(data=np.array(fit.feature_names), name='feature'),
                             table.Column(data=np.array(fit.feature_wavelengths), name='rest_wavelength')])
    hdr['object'] = 'features rest wavelength'
    hdu = fits.table_to_hdu(fit_table)
    hdu.name = 'FEATWL'
    h.append(hdu)

    if is_cube:
        with fits.open(fit.input_data.fitsfile) as original_cube:
            for ext_name in ['vor', 'vorplus']:
                if ext_name in original_cube:
                    h.append(original_cube[ext_name])
            h.writeto(out_image, overwrite=overwrite, checksum=True)
    else:
        h.writeto(out_image, overwrite=overwrite, checksum=True)


def table_to_config(t: table.Table):
    cfg = configparser.ConfigParser()
    previous_section = None
    for line in t:
        if line['parameters'].count('.') > 1:
            replace_occurrences = line['parameters'].count('.') - 1
            warnings.warn(f'Found more than one "." in parameter name {line["parameters"]}. '
                          f'Replacing first {replace_occurrences} occurrences by "_".')
            line['parameters'] = line['parameters'].replace('.', '_', replace_occurrences)
        section, option = line['parameters'].split('.')
        if option in ['h3', 'h4']:
            warnings.warn(f'Converting parameter name {option} to {"_".join(option)}')
            option = '_'.join(option)
        value = line['values']
        if section != previous_section:
            cfg.add_section(section)
            previous_section = section
        sec = cfg[section]
        sec[option] = value
    return cfg


def features_from_table(fit, parameter_names, solution, rest_wavelength):
    features = {}
    feature_wavelength = dict(rest_wavelength)
    for i, j in enumerate(parameter_names):
        name, parameter = j
        if name in features:
            features[name][parameter] = solution[i]
        else:
            features[name] = {'name': name, 'rest_wavelength': feature_wavelength[name], parameter: solution[i]}
    for name in features:
        fit.add_feature(**features[name])
    fit.pack_groups()
    return fit


def load_fit(file_name):
    """
    Loads the result of a previous fit, and put it in the
    appropriate variables for the plot function.

    Parameters
    ----------
    file_name : string
        Name of the FITS file containing the fit results.

    Returns
    -------
    Nothing.
    """
    with fits.open(file_name, mode='readonly') as h:
        if len(h['FITSPEC'].data.shape) == 3:
            data = datacube.Cube(file_name, scidata='FITSPEC', variance='VAR', flags='FLAGS', stellar='STELLAR',
                                 primary='PRIMARY', spatial_mask='MASK2D', redshift=0.0)
        # Redshift is set to zero because LineFit already puts everything in the rest frame.
        elif len(h['FITSPEC'].data.shape) == 1:
            data = onedspec.Spectrum(file_name, scidata='FITSPEC', variance='VAR', flags='FLAGS', stellar='STELLAR',
                                     primary='PRIMARY', redshift=0.0)
        else:
            raise RuntimeError(
                f'Data dimensions are expected to be either 1 or 3, got "{len(h["FITSPEC"].data.shape)}".')

        if 'FITCONFIG' in h:
            config = table_to_config(table.Table(h['FITCONFIG'].data))
            line_fit_config = parser.LineFitParser(config).get_vars()
            line_fit_config['refit'] = False
            fit = setup_fit(data, **line_fit_config)
            fit.pack_groups()
        else:
            if len(h['FITSPEC'].data.shape) == 3:
                fit = modeling.LineFit3D(data)
            elif len(h['FITSPEC'].data.shape) == 1:
                fit = modeling.LineFit(data)
            fit = features_from_table(fit, parameter_names=h['parnames'].data, solution=h['solution'].data,
                                      rest_wavelength=h['featwl'].data)

        translate_extensions = {'pseudo_continuum': 'fitcont', 'model': 'model', 'eqw_model': 'eqw_m',
                                'eqw_direct': 'eqw_d', 'flux_model': 'flux_m', 'flux_direct': 'flux_d',
                                'status': 'status', 'reduced_chi_squared': 'red_chi'}
        # Backwards compatibility
        key = 'solution'
        if len(fit.parameter_names) == (len(h[key].data) - 1):
            warnings.warn('It seems you are trying to read a file from IFSCube v1.0. '
                          'Removing last plane from solution extension.', stacklevel=2)
            setattr(fit, key, h[key].data[:-1])
        else:
            setattr(fit, key, h[key].data)
        for key in translate_extensions:
            value = translate_extensions[key]
            if value in h:
                setattr(fit, key, h[translate_extensions[key]].data)
            else:
                warnings.warn(
                    f'Extension {value} not found in file {h.filename()}. Adding place holder data to this extension.')
                if key == 'reduced_chi_squared':
                    setattr(fit, key, np.zeros(data.data.shape[1:]))
                else:
                    warnings.warn(f'No behaviour set for extension {key}. Leaving empty.')

    return fit
