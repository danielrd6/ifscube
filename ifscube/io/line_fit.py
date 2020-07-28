import numpy as np
from astropy import table
from astropy.io import fits


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
    assert np.all(np.abs((d_wl / avg_d_wl) - 1)) < 1e-6, 'Wavelength vector is not linearly sampled.'

    hdr['CD1_1'] = (avg_d_wl, 'CD1_1')
    hdu = fits.ImageHDU(data=fit.data, header=hdr)
    hdu.name = 'FITSPEC'
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
    hdu = fits.ImageHDU(data=spectrum.resultspec, header=hdr)
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

    if spectrum.fit_dispersion is not None:
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
    hdu = fits.ImageHDU(data=spectrum.eqw_model, header=hdr)
    hdu.name = 'EQW_M'
    h.append(hdu)

    hdr['object'] = 'eqw_direct'
    hdu = fits.ImageHDU(data=spectrum.eqw_direct, header=hdr)
    hdu.name = 'EQW_D'
    h.append(hdu)

    # # Creates the minimize's exit status extension
    # hdr['object'] = 'status'
    # hdu = fits.ImageHDU(data=spectrum.fit_status, header=hdr)
    # hdu.name = 'STATUS'
    # h.append(hdu)

    # Creates component and parameter names table.
    fit_table = table.Table(np.array(fit.parameter_names), names=('component', 'parameter'))
    hdr['object'] = 'parameter names'
    hdu = fits.table_to_hdu(fit_table)
    hdu.name = 'PARNAMES'
    h.append(hdu)

    h.writeto(out_image, overwrite=args['overwrite'])
