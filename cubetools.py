#!/usr/bin/env python

from numpy import *
import pyfits as pf
import spectools as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from copy import deepcopy
from voronoi_2d_binning import voronoi_2d_binning
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter as gf
from scipy.integrate import trapz


# Defining a few function that will be needed later

linename = array(['[N II]', 'Ha', '[N II]', '[S II]', '[S II]'])
lc = array([6548.04, 6562.80, 6583.46, 6716.44, 6730.82])
#hanii_model = lambda x,z,a1,a2,a3,s : a1*exp(-(x-lc[0]*(1.+z))**2/2./s**2) \
#                                    + a2*exp(-(x-lc[1]*(1.+z))**2/2./s**2) \
#                                    + a3*exp(-(x-lc[2]*(1.+z))**2/2./s**2)

hanii_model = lambda x,v,a1,a2,a3,s : \
      a1*exp(-(x-lc[0]*(1.+v/2.998e+5))**2/2./s**2) \
    + a2*exp(-(x-lc[1]*(1.+v/2.998e+5))**2/2./s**2) \
    + a3*exp(-(x-lc[2]*(1.+v/2.998e+5))**2/2./s**2)

haniisii_model = lambda x,v,a1,a2,a3,a4,a5,s : \
    + a1*exp(-(x-lc[0]*(1.+v/2.998e+5))**2/2./s**2) \
    + a2*exp(-(x-lc[1]*(1.+v/2.998e+5))**2/2./s**2) \
    + a3*exp(-(x-lc[2]*(1.+v/2.998e+5))**2/2./s**2) \
    + a4*exp(-(x-lc[3]*(1.+v/2.998e+5))**2/2./s**2) \
    + a5*exp(-(x-lc[4]*(1.+v/2.998e+5))**2/2./s**2)

def progress(x,xmax,steps=10):
    if x%(xmax/steps) == 0:
        print '{:2.0f}%\r'.format(float(x)/float(xmax)*100)

class gmosdc:
    """
    A class for dealing with data cubes, originally written to
    work with GMOS IFU.
    """
  
    def __init__(self, fitsfile, redshift=None, vortab=None):
        """
        Initializes the class and loads basic information onto the object.
    
        Parameters:
        -----------
        fitstile : string
            Name of the FITS file containing the GMOS datacube. This should
            be the standard output from the GFCUBE task of the GEMINI-GMOS
            IRAF package.
        redshift : float
            Value of redshift (z) of the source, if no Doppler correction has
            been applied to the spectra yet.
    
    
        Returns:
        --------
        Nothing.
        """
    
        if len(pf.open(fitsfile)) == 2:
            dataext, hdrext = 1,0
        elif len(pf.open(fitsfile)) == 1:
            dataext, hdrext = 0,0

        self.data = pf.getdata(fitsfile,ext=dataext)
        self.header_data = pf.getheader(fitsfile, ext=dataext)
        self.header = pf.getheader(fitsfile, ext=hdrext)

        self.wl = st.get_wl(fitsfile, hdrext=dataext, dimension=0,
            dwlkey='CD3_3', wl0key='CRVAL3', pix0key='CRPIX3')

        if redshift == None:
            try:
                redshift = self.header['REDSHIFT']
            except KeyError:
                print 'WARNING! Redshift not given and not found in the image'\
                    + ' header. Using redshift = 0.'
                redshift = 0.0
        self.restwl = self.wl/(1.+redshift)

        try:
            if self.header['VORBIN'] and vortab != None:
                self.voronoi_tab = vortab
                self.binned = True
            elif self.header['VORBIN'] and vortab == None:
                print 'WARNING! Data has been binned but no binning table has'\
                    + ' been given.'
                self.binned = True
        except KeyError:
            self.binned = False

        self.fitsfile = fitsfile
        self.redshift = redshift
        self.spec_indices = column_stack([
            ravel(indices(shape(self.data)[1:])[0]),
            ravel(indices(shape(self.data)[1:])[1])
            ])
    
    def continuum(self, niterate=3, degr=5, upper_threshold=1,
        lower_threshold=1, writefits=False, outimage=None,
        fitting_window=None):
        """
        Evaluates a polynomial continuum for the whole cube
        and stores it in self.continuum.
        """

        if self.binned:
            v = loadtxt(self.voronoi_tab)
            xy = v[unique(v[:,2],return_index=True)[1],:2]
        else:
            xy = self.spec_indices


        c = zeros(shape(self.data))
        wl = self.restwl        
        nspec = len(xy)

        for k,h in enumerate(xy):
            if k%(nspec/100) == 0:
                print '{:02d}%\r'.format(k/(nspec/100))
            i,j = h
            s = deepcopy(self.data[:,i,j])
            if any(s[:20]):
                try:
                    cont = st.continuum(wl, s, niterate=niterate,
                        degr=degr, upper_threshold=upper_threshold,
                        lower_threshold=lower_threshold, returns='function')[1]
                    if self.binned:
                        for l,m in v[v[:,2] == k,:2]:
                            c[:,l,m] = cont
                    else:
                        c[:,i,j] = cont
                except TypeError:
                    print 'Could not find a solution for {:d},{:d}.'\
                        .format(i,j)
                    return wl, s

            else:
                c[:,i,j] = zeros(len(wl))
    
        self.cont = c
    
        if writefits:
            if outimage == None:
                outimage = self.fitsfile.replace('.fits','_continuum.fits')
    
            hdr = deepcopy(self.header_data)
    
            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr.append(('REDSHIFT',self.redshift,'Redshift used in GMOSDC'))
    
            hdr.append(('CONTDEGR',degr,'Degree of continuum polynomial'))
            hdr.append(('CONTNITE',niterate,'Continuum rejection iterations'))
            hdr.append(('CONTLTR',lower_threshold,'Continuum lower threshold'))
            hdr.append(('CONTHTR',upper_threshold,'Continuum upper threshold'))
    
            pf.writeto(outimage, data=array([c, self.data-c]), header=hdr)
    
    def snr_eval(self,wl_range=[6050,6200]):
      """
      Measures the signal to noise ratio (SNR) for each spectrum in a data cube,
      returning an image of the SNR.
  
      Parameters:
      -----------
      self : gmosdc instance
        gmosdc object
      wl_range : array like
        An array like object containing two wavelength coordinates that
        define the SNR window at the rest frame.
  
      Returns:
      --------
      snr : numpy.ndarray
        Image of the SNR for each spectrum.
  
      Description:
      ------------
        This method evaluates the SNR for each spectrum in a data cube by
        measuring the residuals of a polynomial continuum fit. The function
        CONTINUUM of the SPECTOOLS package is used to provide the continuum,
        with zero rejection iterations and a 3 order polynomial.
      """
  
      noise = zeros(shape(self.data)[1:])
      signal = zeros(shape(self.data)[1:])
  
      snrwindow = (self.restwl >= wl_range[0]) & (self.restwl <= wl_range[1])
  
      try:
        d = self.data - self.cont
      except AttributeError:
        self.continuum()
        d = self.data - self.cont
  
      for i,j in self.spec_indices:
        if any(d[snrwindow,i,j]):
          noise[i,j] = std(d[snrwindow,i,j])
          signal[i,j] = average(self.cont[snrwindow,i,j])
        else:
          noise[i,j],signal[i,j] = nan,nan
      
      self.noise = noise
      self.signal = signal
  
      return array([signal,noise])
  
    def wlprojection(self, wl0, fwhm=10, filtertype='box', writefits=False,
        outimage='wlprojection.fits'):
      """
      Writes a projection of the data cube along the wavelength coordinate,
      with the flux given by a given type of filter.
  
      Parameters:
      -----------
      wl0 : float
        Central wavelength at the rest frame.
      fwhm : float
        Full width at half maximum. See 'filtertype'.
      filtertype : string
        Type of function to be multiplied by the spectrum to return the
        argument for the integral.
        'box'      = Box function that is zero everywhere and 1 between
                     wl0-fwhm/2 and wl0+fwhm/2.
        'gaussian' = Normalized gaussian function with center at wl0 and
                     sigma = fwhm/(2*sqrt(2*log(2)))
      outimage : string
        Name of the output image
  
      Returns:
      --------
      Nothing.
      """
      
      if filtertype == 'box':
         arrfilt = array( (self.restwl >= wl0-fwhm/2.) & 
                          (self.restwl <= wl0+fwhm/2.), dtype='float')
         arrfilt /= trapz(arrfilt,self.restwl)
      elif filtertype == 'gaussian':
         s = fwhm/(2.*sqrt(2.*log(2.)))
         arrfilt = 1./sqrt(2*pi)*exp(-(self.restwl-wl0)**2/2./s**2)
      else:
        print 'ERROR! Parameter filtertype "{:s}" not understood.'\
            .format(filtertype)
  
      outim = zeros(shape(self.data)[1:])
  
      for i,j in self.spec_indices:
        outim[i,j] = trapz(self.data[:,i,j]*arrfilt,self.restwl)
  
      if writefits:
  
        hdr = deepcopy(self.header)
  
        try:
            hdr['REDSHIFT'] = self.redshift
        except KeyError:
            hdr.append(('REDSHIFT', self.redshift, 'Redshift used in GMOSDC'))
        hdr.append(('WLPROJ', True, 'Processed by WLPROJECTION?'))
        hdr.append(('WLPRTYPE', filtertype,
            'Type of filter used in projection.'))
        hdr.append(('WLPRWL0', wl0, 'Central wavelength of the filter.'))
        hdr.append(('WLPRFWHM', fwhm, 'FWHM of the projection filter.'))
  
        pf.writeto(outimage,data=outim,header=hdr)      
  
      return outim
    
    def plotspec(self,x,y):
      """
      Plots the spectrum at coordinates x,y.
      """
  
      fig = plt.figure(1)
      ax = plt.axes()
  
      try:
        if len(x) == 2 and len(y) == 2:
          s = average(average(self.data[:,y[0]:y[1],x[0]:x[1]],1),1)
      except TypeError:
          s = self.data[:,y,x]
  
      ax.plot(self.restwl,s)
  
      plt.show()
  
    def linefit(self, p0=[1e-16, 6563, 1.5], function='gaussian',
            fitting_window=None, writefits=False, outimage=None, snr=10,
            scale_factor=1, c_niterate=3, c_degr=7, c_upper_threshold=3,
            c_lower_threshold=5, cf_sigma=None, inst_disp=1.0, bounds=None):
        """
        Fits a spectral feature with a gaussian function and returns a
        map of measured properties.
    
        Parameters
        ----------
        p0 : iterable
            Initial guess for the fitting funcion. If scale_factor is
            set to something different than 1, the initial guess must
            be changed accordingly.
        function : string
            The function to be fitted to the spectral features.
            Available options and respective parameters are:
                'gaussian' : amplitude, central wavelength in angstroms,
                    sigma in angstroms
                '2gaussian' : the same as the above, only repeated two
                    times
                '3gaussian' : the same as the above, only repeated 
                    three times
        fitting_window : iterable
            Lower and upper wavelength limits for the fitting
            algorithm. The wavelength coordinates refer to the rest
            frame.
        writefits : boolean
            Writes the results in a FITS file.
        outimage : string
            Name of the FITS file in which to write the results.
        snr : float
            Estimate of signal to noise ratio. This is only used to
            evaluate the reduced chi squared of the fits.
        scale_factor : float
            A scale factor to be applied to the flux of the spectra.
            If the flux values are too small the minimizing algorithm
            has difficulties with the convergence criteria. Choose a
            scale factor that approximates the flux to one.
        c_niterate : integer
            Number of continuum rejection iterations.
        c_degr : integer
            Order of the polynomial for continuum fitting.
        c_upper_threshold : integer
            Upper threshold for continuum fitting rejection in units of
            standard deviations.
        c_lower_threshold : integer
            Lower threshold for continuum fitting rejection in units of
            standard deviations.
        inst_disp : number
            Instrumental dispersion in pixel units. This argument is
            used to evaluate the reduced chi squared. If let to default
            it is assumed that each wavelength coordinate is a degree
            of freedom. The physically sound way to do it is to use the
            number of dispersion elements in a spectrum as the degrees
            of freedom.
        bounds : sequence
            Bounds for the fitting algorithm, given as a sequence of
            (xmin, xmax) pairs for each parameter.
           
        Returns
        -------
        sol : numpy.ndarray
            A data cube with the solution for each spectrum occupying
            the respective position in the image, and each position in
            the first axis giving the different parameters of the fit.

        See also
        --------
        scipy.optimize.curve_fit, scipy.optimize.leastsq
        """

        if function == 'gaussian':
            fit_func = lambda x, a ,b, c: a*exp(-(x-b)**2/2./c**2)
            self.fit_func = fit_func
        if function == '2gaussian':
            fit_func = lambda x, a1, b1, c1, a2, b2, c2:\
                a1*exp(-(x-b1)**2/2./c1**2) + a2*exp(-(x-b2)**2/2./c2**2)
            self.fit_func = fit_func
        if function == '3gaussian':            
            fit_func = lambda x, a1, b1, c1, a2, b2, c2, a3, b3, c3:\
                a1*exp(-(x-b1)**2/2./c1**2) + a2*exp(-(x-b2)**2/2./c2**2)\
                + a3*exp(-(x-b3)**2/2./c3**2)
            self.fit_func = fit_func
        
        if fitting_window != None:
            fw = (self.restwl > fitting_window[0]) &\
                 (self.restwl < fitting_window[1])
            wl = deepcopy(self.restwl[fw])
            data = deepcopy(self.data[fw,:,:])*scale_factor
        else:
            wl = deepcopy(self.restwl)
            data = deepcopy(self.data)*scale_factor
        npars = len(p0)
        nan_solution = array([nan for i in range(npars+1)])
        sol = zeros((npars+1,shape(self.data)[1], shape(self.data)[2]))
        self.fitcont = zeros(shape(data))
        self.fitwl = wl
        self.fitspec = zeros(shape(data))

        if self.binned:
            v = loadtxt(self.voronoi_tab)
            xy = v[unique(v[:,2],return_index=True)[1],:2]
        else:
            xy = self.spec_indices
        
        # Scale factor for the flux. Needed to avoid problems with
        # the minimization algorithm.
        flux_sf = ones(npars)
        flux_sf[arange(0,npars,3)] *= scale_factor
        nspec = len(xy)
        for k,h in enumerate(xy):
            progress(k,nspec,10)
            i,j = h
            s = data[:,i,j]
            cont = st.continuum(wl, s, niterate=c_niterate,
                   degr=c_degr, upper_threshold=c_upper_threshold,
                   lower_threshold=c_lower_threshold, returns='function')[1]
            s = data[:,i,j] - cont
            # Avoids fitting if the spectrum is null.
            if ~any(s[:20]):
                sol[:,i,j] = nan_solution
                continue
            try:
                res = lambda x : sum(abs(fit_func(self.fitwl,*x) - s))
                r = minimize(res, x0=p0, method='SLSQP', bounds=bounds)
                # Reduced chi squared of the fit.
                chi2 = sum(((fit_func(self.fitwl,*r['x'])-s)/s/snr)**2)
                nu = len(s)/inst_disp - npars - 1
                red_chi2 = chi2 / nu
                p = append(r['x']/flux_sf,red_chi2)
            except RuntimeError:
                print 'Optimal parameters not found for spectrum {:d},{:d}'\
                    .format(int(i),int(j))
                p = nan_solution
            if self.binned:
                for l,m in v[v[:,2] == k,:2]:
                    sol[:,l,m] = p
                    self.fitcont[:,l,m] = cont/scale_factor
                    self.fitspec[:,l,m] = s/scale_factor
            else:
                sol[:,i,j] = p
                self.fitcont[:,i,j] = cont/scale_factor
                self.fitspec[:,i,j] = s/scale_factor

        self.em_model = sol
    
        if writefits:
            if outimage == None:
                outimage = self.fitsfile.replace('.fits',
                    '_linefit.fits')
            hdr = deepcopy(self.header_data)
            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr.append(('REDSHIFT', self.redshift, 
                    'Redshift used in GMOSDC'))
            hdr.append(('SLICE0','Angstroms','Central wavelength'))
            hdr.append(('SLICE1','erg/cm2/s/A/arcsec2','Amplitude'))
            hdr.append(('SLICE2','Angstroms','Sigma'))
            pf.writeto(outimage,data=sol,header=self.header_data)
        return sol

    def eqw(self, amp_index=1, sigma_index=2, sigma_limit=3):
        """
        Evaluates the equivalent width of a previous linefit.
        """
        xy = self.spec_indices
        eqw = zeros(shape(self.em_model)[1:])
        
        for i,j in xy:
            cond = (self.fitwl > self.em_model[amp_index,i,j]\
                - sigma_limit*self.em_model[sigma_index,i,j])\
                & (self.fitwl < self.em_model[amp_index,i,j]\
                + sigma_limit*self.em_model[sigma_index,i,j])
            fit = self.fit_func(self.fitwl[cond], *self.em_model[:-1,i,j])
            cont = self.fitcont[cond,i,j]
            eqw[i,j] = trapz(1. - (fit+cont)/cont, x=self.fitwl[cond])

        return eqw

        
    def plotfit(self, x, y):
        """
        Plots the spectrum and features just fitted.
        """

        fig = plt.figure(1)
        plt.clf()
        ax = plt.axes()

        ax.plot(self.fitwl, self.fit_func(self.fitwl, *self.em_model[:-1,y,x]))
        ax.plot(self.fitwl, self.fitspec[:,y,x])
        print self.em_model[:,y,x]
        plt.show()

    def specmodel(self,x,y,wllim=[6520,6620]):
      """
      Plots the spectrum and model for the emission lines at x,y.
  
      Parameters:
      -----------
      x : float
        Column of the IFU image.
      y : float
        Line of the IFU image.
  
      Returns:
      --------
      m : array
      """
  
      try:
        xyz = shape(self.cont)
      except AttributeError:
        print 'ERROR! This function requires the presence of a continuum \n given as the self.cont attribute. Please run the gmosdc.continuum method \n or assign a previously computed continuum.'
        return 
  
      try:
        xyz = shape(self.em_model)
      except AttributeError:
        print 'ERROR! This function requires the presence of a emission\n \
        model image given as the self.em_model attribute. Please run the\n\
        gmosdc.hafit method or assign a previously computed continuum.'
        return
  
      fig = plt.figure(1)
      plt.clf()
      ax = plt.axes()
  
      cond = (self.restwl > wllim[0])&(self.restwl < wllim[1])
  
      wl,s,c,m = deepcopy(self.restwl[cond]),\
                 deepcopy(self.data[cond,y,x]),\
                 deepcopy(self.cont[cond,y,x]),\
                 deepcopy(self.em_model[:,y,x])
  
      sf = 1e-17         # scale factor for the plots
  
      ax.plot(wl, c/sf, label='Continuum')
      ax.plot(wl, s/sf, label='Observed spectrum')
      if len(m) == 5:
        ax.plot(wl, (hanii_model(wl,*m)+c)/sf, label='Emission model')
      if len(m) == 7:
        ax.plot(wl, (haniisii_model(wl,*m)+c)/sf, label='Emission model')
      ax.minorticks_on()
      ax.set_xlabel(r'Wavelength ($\AA$)')
      ax.set_ylabel(r'Flux density ($10^{{{:d}}}$ erg/s/cm^2/$\AA$)'\
                    .format(int(log10(sf))))
  
      plt.show()
  
    def voronoi_binning(self, targetsnr=10.0, writefits=False,
                        outfile=None, clobber=True, writevortab=True):
        """
        Applies Voronoi binning to the data cube, using Cappellari's
        Python implamentation.
    
        Parameters:
        -----------
        targetsnr : float
          Desired signal to noise ratio of the binned pixels
        writefits : boolean
          Writes a FITS image with the output of the binning.
        outfile : string
          Name of the output FITS file. If 'None' then the name of
          the original FITS file containing the data cube will be used
          as a root name, with '.bin' appended to it.
    
        Returns:
        --------
        Nothing.
        """
    
        try:
            x = shape(self.noise)
        except AttributeError:
            print 'This function requires prior execution of the snr_eval'\
                + 'method.'
            return
    
        x = ravel(indices(shape(self.signal))[0])
        y = ravel(indices(shape(self.signal))[1])
        s, n = deepcopy(self.signal), deepcopy(self.noise)
        s[isnan(s)] = average(self.signal[~isnan(self.signal)])
        s[s <= 0] = average(self.signal[self.signal > 0])
        n[isnan(n)] = average(self.signal[~isnan(self.signal)])*.5
        n[n <= 0] = average(self.signal[self.signal > 0])*.5
        signal, noise = ravel(s),ravel(n)
    
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
            voronoi_2d_binning(x, y, signal, noise, targetsnr, plot=1, quiet=0)
        v = column_stack([x,y,binNum])
    
        if writevortab:
            savetxt('voronoi_binning.dat',v,fmt='%.2f\t%.2f\t%d')
        
        binned = zeros(shape(self.data))
        for i,j in enumerate(v):
            selected = array(v[v[:,2] == v[i,2]][:,:2], dtype='int')
            for l,k in enumerate(selected):
                if l == 0:
                    spec = self.data[:,k[0],k[1]]
                else:
                    spec = row_stack([spec, self.data[:,k[0],k[1]]])
      
            if len(shape(spec)) == 2:
                binned[:,j[0],j[1]] = average(spec,0)
            elif len(shape(spec)) == 1:
                binned[:,j[0],j[1]] = spec
            else:
                print 'ERROR! shape(spec) = {:s}, expecting 1 or 2'\
                    .format(shape(spec))
    
        if writefits:
            hdr = deepcopy(self.header_data)  
            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr.append(('REDSHIFT', self.redshift,
                    'Redshift used in GMOSDC'))
            hdr.append(('VORBIN',True,'Processed by Voronoi binning?'))
            hdr.append(('VORTSNR',targetsnr,'Target SNR for Voronoi binning.'))
            if outfile == None:
                outfile = '{:s}bin.fits'.format(self.fitsfile[:-4])  
            pf.writeto(outfile,data=binned,header=hdr,clobber=clobber)
  
    def write_binnedspec(self, dopcor=False, writefits=False):
        """
        Writes only one spectrum for each bin in a FITS file.
        """

        xy = self.spec_indices
        unique_indices = xy[unique(self.data[1400,:,:], return_index=True)[1]]

        if dopcor:

            try:
                shape(self.em_model)
            except AttributeError:
                print 'ERROR! This function requires the gmosdc.em_model'\
                    + ' attribute to be defined.'
                return

            for k,i,j in enumerate(unique_indices):
                z = self.em_model[0,i,j]/2.998e+5
                interp_spec = interp1d(self.restwl/(1.+z),self.data[i,j])
                if k == 0:
                    specs = interp_spec(self.restwl)
                else:
                    specs = row_stack([specs,interp_spec(self.restwl)])

        else:
            specs = row_stack([self.data[:,i,j] for i,j in unique_indices])

        return specs        
  
    def lineflux(self,amplitude,sigma):
      """
      Calculates the flux in a line given the amplitude and sigma
      of the gaussian function that fits it.
  
      """
  
      lf = amplitude * abs(sigma) * sqrt(2.*pi)
  
      return lf
