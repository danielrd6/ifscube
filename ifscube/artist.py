import numpy as np
from astropy.io import fits

from ifscube import elprofile


def fake_cube():
    wavelength = np.arange(6000, 7000)
    flux = np.ones((len(wavelength), 30, 30)) * 1e-15
    it = np.column_stack([np.ravel(_) for _ in np.indices(flux.shape[1:])])

    for y, x in it:
        flux[:, y, x] += elprofile.gaussvel(
            wavelength, np.array([6562.8]), np.array([1e-13, ((y + x) * 10) - 300, 200]))

    hdu = fits.HDUList()
    hdu.append(fits.PrimaryHDU())

    h = fits.Header()
    for i in range(1, 4):
        h[f'CD{i}_{i}'] = 1
        h[f'CRPIX{i}'] = 1

    h['CRVAL1'] = 0
    h['CRVAL2'] = 0
    h['CRVAL3'] = 4500

    hdu.append(fits.ImageHDU(data=flux, header=h))
    hdu[0].name = 'PRIMARY'
    hdu[1].name = 'SCI'
    hdu.writeto('fake.fits', overwrite=True)


if __name__ == '__main__':
    fake_cube()
