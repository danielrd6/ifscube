from astropy.io import fits
import numpy as np

from . import onedspec


class Spectrum(onedspec.Spectrum):
    """
    Subclass of onedspec.Spectrum, with a _load method appropriate for the
    one dimensional spectrum files of SDSS. The format of the input file
    is explained in
    https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
    """

    def _load(self, fname: str, scidata: str = 'SCI', variance: str = None, flags: str = None, stellar: str = None,
              primary: str = 'PRIMARY', redshift: float = None, wcs_axis: int = None) -> None:

        self.fitsfile = fname
        with fits.open(fname) as hdu:
            self.data = hdu['coadd'].data['flux'] * 1e-17
            self.header = hdu[primary].header
            self.header_data = hdu['coadd'].header

            self.variance = 1. / hdu['coadd'].data['ivar'] * 1e-34
            self.flags = hdu['coadd'].data['ivar'] == 0.0
            self.stellar = np.zeros_like(self.data)
            self.wl = 10 ** hdu['coadd'].data['loglam']
            self.delta_lambda = np.mean(np.diff(self.wl))

            if redshift is not None:
                self.redshift = redshift
            else:
                self.redshift = hdu['specobj'].data['z'][0]

            self.rest_wavelength = self.dopcor(self.redshift, self.wl)
