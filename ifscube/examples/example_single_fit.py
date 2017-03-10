import ifscube.onedspec as ds

x = ds.Spectrum('ngc6300_nuc.fits')
x.linefit([1e-15, 6604, 3, 0, 0], fitting_window=(6500, 6700),
    function='gauss_hermite')
x.plotfit()
