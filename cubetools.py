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
from scipy.ndimage import gaussian_filter as gf


# Defining a few function that will be needed later

linename = array(['[N II]', 'Ha', '[N II]', '[S II]', '[S II]'])
lc = array([6549.85, 6564.61, 6585.28, 6718.29, 6732.67])
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

#def haniisii_model(x,v,a1,a2,a3,a4,a5,s):
#    if s > 3:
#       return 1e+9
#    if any(array([a1, a2, a3, a4, a5]) < 0):
#       return 1e+9
#    else:
#       return a1*exp(-(x-lc[0]*(1.+v/2.998e+5))**2/2./s**2) \
#           + a2*exp(-(x-lc[1]*(1.+v/2.998e+5))**2/2./s**2) \
#           + a3*exp(-(x-lc[2]*(1.+v/2.998e+5))**2/2./s**2) \
#           + a4*exp(-(x-lc[3]*(1.+v/2.998e+5))**2/2./s**2) \
#           + a5*exp(-(x-lc[4]*(1.+v/2.998e+5))**2/2./s**2)


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


        c = zeros(shape(data))
        for k,h in enumerate(xy):
            i,j = h
            s = data[:,i,j]
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
    
            pf.writeto(outimage, data=array([c, data-c]), header=hdr)
    
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
          hdr.append(('REDSHIFT',self.redshift,'Redshift used in GMOSDC'))
  
        hdr.append(('WLPROJ',True,'Processed by WLPROJECTION?'))
        hdr.append(('WLPRTYPE',filtertype,'Type of filter used in projection.'))
        hdr.append(('WLPRWL0',wl0,'Central wavelength of the filter.'))
        hdr.append(('WLPRFWHM',fwhm,'FWHM of the projection filter.'))
  
        pf.writeto(outimage,data=outim,header=hdr)      
  
      return outim
  
    def linefit(self,p0,function='gaussian',fitting_window=[6300,6700]):
      """
      Fits emission lines with a given function and returns a map
      of measured properties.
  
      Parameters:
      -----------
      function : string
        The function to be fitted to the spectral features.
      fittingwindow : iterable
        Lower and upper wavelength limits for the fitting algorithm.
      """
  
      h = zeros((4,shape(self.data)[1],shape(self.data)[2]))
      fw = (self.restwl > fitting_window[0])&(self.restwl > fitting_window[0])
      wl = self.restwl[fw] 
    
      d = self.data - self.cont
  
      for i,j in self.spec_indices:
        s = d[fw,i,j]
        if ~any(s[:20]):
          h[:,i,j] = zeros(4)
          continue
  
        try:
          f,p = st.fitgauss(wl,s,p0,fitcenter=True,fitbg=True)
        except RuntimeError:
          print 'Optimal parameters not found for spectrum {:d},{:d}'\
              .format(i,j)
          p = zeros(4)
  
        h[:,i,j] = p
  
      return h
    
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
  
    def hafit(self, function='gaussian', p0=[50.0,1e-16,1e-16,1e-16,2], 
              fit_window=[6400,6700], writefits=False, outimage=None,
              trysmooth=False):
      """
      Fits the three emission lines centred in H alpha (6564 A) a given 
      function and returns a map of measured properties.
  
      Parameters:
      -----------
      function : string
        The function to be fitted to the spectral features.
      fit_window : iterable
        Lower and upper wavelength limits for the fitting algorithm. The
        wavelength coordinates refer to the rest frame.
      """
  
      fw = (self.restwl > fit_window[0])&(self.restwl < fit_window[1])
      wl = deepcopy(self.restwl[fw])
    
      try:
        x = shape(self.cont)
      except AttributeError:
        print 'This function requires prior execution of the continuum method.'
        return
  
      d = self.data - self.cont
      sol = zeros((5,shape(self.data)[1], shape(self.data)[2]))
  
      lc = array([6549.85, 6564.61, 6585.28])
  
      for i,j in self.spec_indices:
  
        s = d[fw,i,j]
 
        if ~any(s[:20]):
          sol[:,i,j] = zeros(5)
          continue
 
        try:
          p = curve_fit(hanii_model,wl,s,p0=p0)[0]
          if any(p[1:-1] < 0):
            if trysmooth:
              p = curve_fit(hanii_model,wl,gf(s,3),p0=p0)[0]
              
        except RuntimeError:
          print 'Optimal parameters not found for spectrum {:d},{:d}'.format(i,j)
          sol[:,i,j] = zeros(5)
 
        sol[:,i,j] = p
  
      sol[-1] = abs(sol[-1])
      self.em_model = sol
  
      if writefits:
        if outimage == None:
          outimage = self.fitsfile.replace('.fits','_hafit.fits')
  
        hdr = deepcopy(self.header_data)
  
        try:
          hdr['REDSHIFT'] = self.redshift
        except KeyError:
          hdr.append(('REDSHIFT',self.redshift,'Redshift used in GMOSDC'))
  
  
        hdr.append(('SLICE0','km/s','Radial velocity'))
        hdr.append(('SLICE1','erg/cm2/s/A/arcsec2','[N II] {:.2f}'.format(lc[0])))
        hdr.append(('SLICE2','erg/cm2/s/A/arcsec2','H alpha {:.2f}'.format(lc[1])))
        hdr.append(('SLICE3','erg/cm2/s/A/arcsec2','[N II] {:.2f}'.format(lc[2])))
        hdr.append(('SLICE4','Angstroms','Sigma'))
  
        pf.writeto(outimage,data=sol,header=self.header_data)
  
      return sol
  
    def haniisiifit(self, function='gaussian',
            p0=[50, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 2],
            fitting_window=None, writefits=False, outimage=None,
            trysmooth=False, c_niterate=3, c_degr=7,
            c_upper_threshold=3, c_lower_threshold=5,
            cf_sigma=None):
        """
        Fits the three emission lines centred in H alpha (6564 A) a given 
        function and returns a map of measured properties.
    
        Parameters:
        -----------
        function : string
          The function to be fitted to the spectral features.
        fit_window : iterable
          Lower and upper wavelength limits for the fitting algorithm. The
          wavelength coordinates refer to the rest frame.
        """
 
        if fitting_window != None:
            fw = (self.restwl > fitting_window[0]) &\
                 (self.restwl > fitting_window[0])
            wl = deepcopy(self.restwl[fw])
            data = deepcopy(self.data[fw,:,:])
        else:
            wl = deepcopy(self.restwl)
            data = deepcopy(self.data)

        sol = zeros((7,shape(self.data)[1], shape(self.data)[2]))
        self.fitcont = zeros(shape(data))
        self.fitwl = wl
        self.fitspec = zeros(shape(data))

        if self.binned:
            v = loadtxt(self.voronoi_tab)
            xy = v[unique(v[:,2],return_index=True)[1],:2]
        else:
            xy = self.spec_indices

        lc = array([6549.85, 6564.61, 6585.28, 6718.29, 6732.67])
    
        for k,h in enumerate(xy):
            i,j = h
            s = data[:,i,j]
            cont = st.continuum(wl, s, niterate=c_niterate,
                   degr=c_degr, upper_threshold=c_upper_threshold,
                   lower_threshold=c_lower_threshold, returns='function')[1]
            s = data[:,i,j] - cont

            #s[s < 0] = 0

            if ~any(s[:20]):
                sol[:,i,j] = zeros(7)
                continue
    
            try:
                p = curve_fit(haniisii_model, wl, s, p0=p0, sigma=cf_sigma)[0]
            except RuntimeError:
                print 'Optimal parameters not found for spectrum {:d},{:d}'\
                    .format(int(i),int(j))

                p = zeros(7)

            if self.binned:
                for l,m in v[v[:,2] == k,:2]:
                    sol[:,l,m] = p
                    self.fitcont[:,l,m] = cont
                    self.fitspec[:,l,m] = s
            else:
                sol[:,i,j] = p
                self.fitcont[:,i,j] = cont
                self.fitspec[:,i,j] = s  

        sol[-1] = abs(sol[-1])
        self.em_model = sol

    
        if writefits:
          if outimage == None:
            outimage = self.fitsfile.replace('.fits','_haniisiifit.fits')
    
          hdr = deepcopy(self.header_data)
    
          try:
            hdr['REDSHIFT'] = self.redshift
          except KeyError:
            hdr.append(('REDSHIFT',self.redshift,'Redshift used in GMOSDC'))
    
          hdr.append(('SLICE0','km/s','Radial velocity'))
          hdr.append(('SLICE1','erg/cm2/s/A/arcsec2','[N II] {:.2f}'.format(lc[0])))
          hdr.append(('SLICE2','erg/cm2/s/A/arcsec2','H alpha {:.2f}'.format(lc[1])))
          hdr.append(('SLICE3','erg/cm2/s/A/arcsec2','[N II] {:.2f}'.format(lc[2])))
          hdr.append(('SLICE4','erg/cm2/s/A/arcsec2','[S II] {:.2f}'.format(lc[3])))
          hdr.append(('SLICE5','erg/cm2/s/A/arcsec2','[S II] {:.2f}'.format(lc[4])))
          hdr.append(('SLICE6','Angstroms','Sigma'))
    
          pf.writeto(outimage,data=sol,header=self.header_data)
    
        return sol
  
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
        print 'This function requires prior execution of the snr_eval method.'
        return
  
      x, y = ravel(indices(shape(self.signal))[0]),ravel(indices(shape(self.signal))[1])
  
      s,n = deepcopy(self.signal),deepcopy(self.noise)
  
      s[isnan(s)] = average(self.signal[~isnan(self.signal)])
      s[s <= 0] = average(self.signal[self.signal > 0])
  
      n[isnan(n)] = average(self.signal[~isnan(self.signal)])*.5
      n[n <= 0] = average(self.signal[self.signal > 0])*.5
  
      signal, noise = ravel(s),ravel(n)
  
      binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
          x, y, signal, noise, targetsnr, plot=1, quiet=0)
      
      v = column_stack([x,y,binNum])
  
      if writevortab:
        savetxt('voronoi_binning.dat',v,fmt='%.2f\t%.2f\t%d')
  
      binned = zeros(shape(self.data))
  
      for i,j in enumerate(v):
        selected = array(v[v[:,2] == v[i,2]][:,:2],dtype='int')
        for l,k in enumerate(selected):
          if l == 0:
            spec = self.data[:,k[0],k[1]]
          else:
            spec = row_stack([spec,self.data[:,k[0],k[1]]])
  
        if len(shape(spec)) == 2:
          binned[:,j[0],j[1]] = average(spec,0)
        elif len(shape(spec)) == 1:
          binned[:,j[0],j[1]] = spec
        else:
          print 'The end of time is upon us! shape(spec) = {:s}'.format(shape(spec))
  
      if writefits:
        hdr = deepcopy(self.header_data)
  
        try:
          hdr['REDSHIFT'] = self.redshift
        except KeyError:
          hdr.append(('REDSHIFT',self.redshift,'Redshift used in GMOSDC'))
  
  
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
