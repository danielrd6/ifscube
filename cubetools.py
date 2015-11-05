"""
Functions for the analysis of integral field spectroscopy.

Author: Daniel Ruschel Dutra
Website: https://github.com/danielrd6/ifscube
"""

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
from scipy.interpolate import interp1d
import profiles as lprof

def progress(x,xmax,steps=10):
    try:
        if x%(xmax/steps) == 0:
            print '{:2.0f}%\r'.format(float(x)/float(xmax)*100)
    except ZeroDivisionError:
        pass

class gmosdc:
    """
    A class for dealing with data cubes, originally written to work
    with GMOS IFU.
    """
  
    def __init__(self, fitsfile, redshift=None, vortab=None):
        """
        Initializes the class and loads basic information onto the
        object.
    
        Parameters:
        -----------
        fitstile : string
            Name of the FITS file containing the GMOS datacube. This
            should be the standard output from the GFCUBE task of the
            GEMINI-GMOS IRAF package.
        redshift : float
            Value of redshift (z) of the source, if no Doppler
            correction has
            been applied to the spectra yet.
        vortab : string
            Name of the file containing the Voronoi binning table
    
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
    
    def continuum(self, writefits=False, outimage=None, fitting_window=None,
        copts=None):
        """
        Evaluates a polynomial continuum for the whole cube and stores
        it in self.cont.
        """

        if self.binned:
            v = loadtxt(self.voronoi_tab)
            xy = v[unique(v[:,2], return_index=True)[1],:2]
        else:
            xy = self.spec_indices

        fw = fitting_window
        fwidx = (self.restwl > fw[0]) & (self.restwl < fw[1])
        
        wl = deepcopy(self.restwl[fwidx])
        data = deepcopy(self.data[fwidx])

        c = zeros(shape(data))

        nspec = len(xy)

        if copts == None:
            copts = {'degr':3, 'upper_threshold':2,
                'lower_threshold':2, 'niterate':5}

        try:
            copts['returns']
        except KeyError:
            copts['returns'] = 'function'

        for k,h in enumerate(xy):
            i,j = h
            s = deepcopy(data[:,i,j])
            if any(s[:20]) and any(s[-20:]):
                try:
                    cont = st.continuum(wl, s, **copts)
                    if self.binned:
                        for l,m in v[v[:,2] == k,:2]:
                            c[:,l,m] = cont[1]
                    else:
                        c[:,i,j] = cont[1]
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
                hdr.append(('REDSHIFT', self.redshift,
                    'Redshift used in GMOSDC'))

            hdr['CRVAL3'] = wl[0]
            hdr.append(('CONTDEGR', copts['degr'],
                'Degree of continuum polynomial'))
            hdr.append(('CONTNITE', copts['niterate'],
                'Continuum rejection iterations'))
            hdr.append(('CONTLTR', copts['lower_threshold'],
                'Continuum lower threshold'))
            hdr.append(('CONTHTR', copts['upper_threshold'],
                'Continuum upper threshold'))
    
            pf.writeto(outimage, data=c, header=hdr)

        return c
    
    def snr_eval(self,wl_range=[6050,6200]):
        """
        Measures the signal to noise ratio (SNR) for each spectrum in a
        data cube, returning an image of the SNR.
  
        Parameters:
        -----------
        self : gmosdc instance
            gmosdc object
        wl_range : array like
            An array like object containing two wavelength coordinates
            that define the SNR window at the rest frame.
  
        Returns:
        --------
        snr : numpy.ndarray
            Image of the SNR for each spectrum.
  
        Description:
        ------------
            This method evaluates the SNR for each spectrum in a data
            cube by measuring the residuals of a polynomial continuum
            fit. The function CONTINUUM of the SPECTOOLS package is used
            to provide the continuum, with zero rejection iterations
            and a 3 order polynomial.
        """
  
        noise = zeros(shape(self.data)[1:])
        signal = zeros(shape(self.data)[1:])
        snrwindow = (self.restwl >= wl_range[0]) & (self.restwl <= wl_range[1])
        data = deepcopy(self.data)

        wl = self.restwl[snrwindow]

        for i,j in self.spec_indices:
            if any(data[snrwindow,i,j]):
                s = data[snrwindow,i,j]
                cont = st.continuum(wl, s, niterate=0,
                    degr=3, upper_threshold=3, lower_threshold=3,
                    returns='function')[1]
                noise[i,j] = nanstd(s - cont)
                signal[i,j] = nanmean(cont)
            else:
                noise[i,j],signal[i,j] = nan, nan
        
        self.noise = noise
        self.signal = signal
  
        return array([signal,noise])
  
    def wlprojection(self, wl0, fwhm=10, filtertype='box', writefits=False,
        outimage='wlprojection.fits'):
        """
        Writes a projection of the data cube along the wavelength
        coordinate, with the flux given by a given type of filter.
   
        Parameters:
        -----------
        wl0 : float
            Central wavelength at the rest frame.
        fwhm : float
            Full width at half maximum. See 'filtertype'.
        filtertype : string
            Type of function to be multiplied by the spectrum to return
            the argument for the integral.
            'box'      = Box function that is zero everywhere and 1
                         between wl0-fwhm/2 and wl0+fwhm/2.
            'gaussian' = Normalized gaussian function with center at
                         wl0 and sigma = fwhm/(2*sqrt(2*log(2)))
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
          outim[i,j] = trapz(self.data[:,i,j]*arrfilt, self.restwl)
   
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

        Parameters
        ----------
        x,y : numbers or tuple
            If x and y are numbers plots the spectrum at the specific
            spaxel. If x and y are two element tuples plots the average
            between x[0],y[0] and x[1],y[1]

        Returns
        -------
        Nothing.
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
    
    def linefit(self, p0, function='gaussian', fitting_window=None,
            writefits=False, outimage=None, variance=None,
            constraints=(), bounds=None, inst_disp=1.0, individual_spec=False,
            min_method='SLSQP', minopts=None, copts=None,
            refit=False, spiral_loop=False, spiral_center=None,
            fit_continuum=True):
        """
        Fits a spectral feature with a gaussian function and returns a
        map of measured properties. This is a wrapper for the scipy
        minimize function that basically iterates over the cube,
        has a formula for the reduced chi squared, and applies
        an internal scale factor to the flux.
    
        Parameters
        ----------
        p0 : iterable
            Initial guess for the fitting funcion, consisting of a list
            of 3N parameters for N components of **function**. In the
            case of a gaussian fucntion, these parameters must be given
            as [amplitude0, center0, sigma0, amplitude1, center1, ...].
        function : string
            The function to be fitted to the spectral features.
            Available options and respective parameters are:
                'gaussian' : amplitude, central wavelength in angstroms,
                    sigma in angstroms
                'gauss_hermite' : amplitude, central wavelength in
                    angstroms, sigma in angstroms, h3 and h4
        fitting_window : iterable
            Lower and upper wavelength limits for the fitting
            algorithm. These limits should allow for a considerable
            portion of continuum besides the desired spectral features.
        writefits : boolean
            Writes the results in a FITS file.
        outimage : string
            Name of the FITS file in which to write the results.
        variance : float, 1D, 2D or 3D array
            The variance of the flux measurments. It can be given
            in one of four formats. If variance is a float it is
            applied as a contant to the whole spectrum. If given as 1D
            array it assumed to be a spectrum that will be applied to
            the whole cube. As 2D array, each spaxel will be applied
            equally to all wavelenths. Finally the 3D array must
            represent the variance for each elemente of the data cube.
            It defaults to None, in which case it does not affect the
            minimization algorithm, and the returned Chi2 will be in
            fact just the fit residuals.
        inst_disp : number
            Instrumental dispersion in pixel units. This argument is
            used to evaluate the reduced chi squared. If let to default
            it is assumed that each wavelength coordinate is a degree
            of freedom. The physically sound way to do it is to use the
            number of dispersion elements in a spectrum as the degrees
            of freedom.
        bounds : sequence
            Bounds for the fitting algorithm, given as a list of
            [xmin, xmax] pairs for each x parameter.
        constraints : dict or sequence of dicts
            See scipy.optimize.minimize
        min_method : string
            Minimization method. See scipy.optimize.minimize.
        minopts : dict
            Dictionary of options to be passed to the minimization
            routine. See scipy.optimize.minimize.
        individual_spec : False or x,y pair
            Pixel coordinates for the spectrum you wish to fit
            individually.
        copts : dict
            Arguments to be passed to the spectools.continuum function.
        refit : boolean
            Use parameters from nearby sucessful fits as the initial
            guess for the next fit.
        spiral_loop : boolean
            Begins the fitting with the central spaxel and continues
            spiraling outwards.
        spiral_center : iterable
            Central coordinates for the beginning of the spiral given
            as a list of two coordinates [x0, y0]
        fit_continuum : boolean
            If True fits the continuum just before attempting to fit
            the emission lines. Setting this option to False will
            cause the algorithm to look for self.cont, which should
            contain a data cube of continua.

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
            fit_func = lprof.gauss
            self.fit_func = fit_func
            npars_pc = 3
        elif function == 'gauss_hermite':
            fit_func = lprof.gausshermite
            self.fit_func = fit_func
            npars_pc = 5
        else:
            raise NameError('Unknown function "{:s}".'.format(function))

        if fitting_window != None:
            fw = (self.restwl > fitting_window[0]) &\
                 (self.restwl < fitting_window[1])
        else:
            fw = Ellipsis

        if copts == None:
            copts = {'niterate':5, 'degr':4, 'upper_threshold':2,
                'lower_threshold':2}
        
        copts['returns'] = 'function'

        try:
            minopts['eps']
        except TypeError:
            if minopts == None:
                minopts = {'eps': 1e-3}
            else:
                minopts['eps'] = 1e-3

        wl = deepcopy(self.restwl[fw])
        scale_factor = median(self.data[fw,:,:])
        data = deepcopy(self.data[fw,:,:])/scale_factor
        fit_status = ones(shape(data)[1:])*-1

        if len(shape(variance)) == 0:
            if variance == None:
                variance = 1.0
        else:
            variance = deepcopy(variance)/scale_factor**2

        vcube = ones(shape(data))
        if len(shape(variance)) == 0:
            vcube *= variance
        elif len(shape(variance)) == 1:
            for i,j in self.spec_indices:
                vcube[:,i,j] = variance
        elif len(shape(variance)) == 2:
            for i,j in enumerate(vcube):
                vcube[i] = variance
        elif len(shape(variance)) == 3:
            vcube = variance

        npars = len(p0)
        nan_solution = array([nan for i in range(npars+1)])
        sol = zeros((npars+1,shape(self.data)[1], shape(self.data)[2]))
        self.fitcont = zeros(shape(data))
        self.fitwl = wl
        self.fitspec = zeros(shape(data))
        self.resultspec = zeros(shape(data))

        if self.binned:
            vor = loadtxt(self.voronoi_tab)
            xy = vor[unique(vor[:,2],return_index=True)[1],:2]
        else:
            xy = self.spec_indices
        
        # Scale factor for the flux. Needed to avoid problems with
        # the minimization algorithm.
        flux_sf = ones(npars)
        flux_sf[arange(0, npars, npars_pc)] *= scale_factor
        p0 /= flux_sf
        if bounds != None:
            bounds = array(bounds)
            for i,j in enumerate(bounds):
                j /= flux_sf[i]

        Y, X = indices(shape(data)[1:])

        if individual_spec:
            xy = [individual_spec[::-1]]
        elif spiral_loop:
            y, x = self.spec_indices[:,0], self.spec_indices[:,1]
            if spiral_center == None:
                r = sqrt((x - x.max()/2.)**2 + (y - y.max()/2.)**2)
            else:
                r = sqrt((x - spiral_center[0])**2 + (y - spiral_center[1])**2)
            t = arctan2(y - y.max()/2., x - x.max()/2.)
            t[t < 0] += 2*pi
            
            b = array([(ravel(r)[i], ravel(t)[i]) for i in\
                range(len(ravel(r)))], dtype=[('radius', 'f8'),\
                ('angle', 'f8')])

            s = argsort(b, axis=0, order=['radius', 'angle'])
            xy = column_stack([ravel(y)[s], ravel(x)[s]])

        nspec = len(xy)
        for k, h in enumerate(xy):
            progress(k, nspec, 10)
            i,j = h
            if ~any(data[:20,i,j]) or ~any(data[-20:,i,j]):
                sol[:,i,j] = nan_solution
                continue
            v = vcube[:,i,j]
            if fit_continuum:
                cont = st.continuum(wl, data[:,i,j], **copts)[1]
            else:
                cont = self.cont[:,i,j]/scale_factor
            s = data[:,i,j] - cont

            # Avoids fitting if the spectrum is null.
            try:
                res = lambda x : sum( (s-fit_func(self.fitwl, x))**2/v )

                if refit and k != 0:
                    radsol = sqrt((Y - i)**2 + (X - j)**2)
                    nearsol = sol[:-1, (radsol < 2) & (fit_status == 0)]
                    if shape(nearsol) == (5, 1):
                        p0 = deepcopy(nearsol.transpose()/flux_sf)
                    else:
                        p0 = deepcopy(average(nearsol.transpose(), 0)/flux_sf)

                r = minimize(res, x0=p0, method=min_method, bounds=bounds,
                    constraints=constraints, options=minopts)
                if r.status != 0:
                    print h, r.message
                # Reduced chi squared of the fit.
                chi2 = res(r['x'])
                nu = len(s)/inst_disp - npars - 1
                red_chi2 = chi2 / nu
                p = append(r['x']*flux_sf, red_chi2)
                fit_status[i,j] = r.status
            except RuntimeError:
                print 'Optimal parameters not found for spectrum {:d},{:d}'\
                    .format(int(i),int(j))
                p = nan_solution
            if self.binned:
                for l, m in vor[vor[:,2] == k,:2]:
                    sol[:,l,m] = p
                    self.fitcont[:,l,m] = cont*scale_factor
                    self.fitspec[:,l,m] = (s+cont)*scale_factor
                    self.resultspec[:,l,m] = (cont+fit_func(self.fitwl,
                        r['x']))*scale_factor
            else:
                sol[:,i,j] = p
                self.fitcont[:,i,j] = cont*scale_factor
                self.fitspec[:,i,j] = (s+cont)*scale_factor
                self.resultspec[:,i,j] = (cont+fit_func(self.fitwl, r['x']))\
                    *scale_factor

        self.em_model = sol
        self.fit_status = fit_status
        p0 *= flux_sf
    
        if writefits:
            
            # Basic tests and first header
            if outimage == None:
                outimage = self.fitsfile.replace('.fits',
                    '_linefit.fits')
            hdr = deepcopy(self.header_data)
            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr.append(('REDSHIFT', self.redshift, 
                    'Redshift used in GMOSDC'))

            # Creates MEF output.
            h = pf.HDUList()
            h.append(pf.PrimaryHDU(header=hdr))

            # Creates the fitted spectrum extension
            hdr = pf.Header()
            hdr.append(('object', 'spectrum', 'Data in this extension'))
            hdr.append(('CRPIX3', 1, 'Reference pixel for wavelength'))
            hdr.append(('CRVAL3', wl[0], 'Reference value for wavelength'))
            hdr.append(('CD3_3', average(diff(wl)),
                'CD3_3'))           
            h.append(pf.ImageHDU(data=self.fitspec, header=hdr))

            # Creates the fitted continuum extension.
            hdr['object'] = 'continuum'
            h.append(pf.ImageHDU(data=self.fitcont, header=hdr))

            # Creates the fitted function extension.
            hdr['object'] = 'fit'
            h.append(pf.ImageHDU(data=self.resultspec, header=hdr))

            # Creates the solution extension.
            hdr['object'] = 'parameters'
            hdr.append(('function', function, 'Fitted function'))
            hdr.append(('nfunc', len(p)/3, 'Number of functions'))
            h.append(pf.ImageHDU(data=sol, header=hdr))

            # Creates the minimize's exit status extension
            hdr['object'] = 'status'
            h.append(pf.ImageHDU(data=fit_status, header=hdr))
            
            h.writeto(outimage)

        if individual_spec:
            return wl, s*scale_factor, cont*scale_factor,\
                fit_func(wl, p[:-1]), r
        else:
            return sol

    def loadfit(self, fname):
        """
        Loads the result of a previous fit, and put it in the
        appropriate variables for the plotfit function.

        Parameters
        ----------
        fname : string
            Name of the FITS file generated by gmosdc.linefit.

        Returns
        -------
        Nothing.
        """

        self.fitwl = st.get_wl(fname, pix0key='crpix3', wl0key='crval3',
            dwlkey='cd3_3', hdrext=1, dataext=1)
        self.fitspec = pf.getdata(fname, ext=1)
        self.fitcont = pf.getdata(fname, ext=2)
        self.resultspec = pf.getdata(fname, ext=3)
        
        func_name = pf.getheader(fname, ext=4)['function']
        if func_name == 'gaussian':
            self.fit_func = lprof.gauss
        if func_name == 'gauss_hermite':
            self.fit_func = lprof.gausshermite
        self.em_model = pf.getdata(fname, ext=4)

    def eqw(self, amp_index=0, center_index=1, sigma_index=2, sigma_limit=3):
        """
        Evaluates the equivalent width of a previous linefit.
        """
        xy = self.spec_indices
        eqw_model = zeros(shape(self.em_model)[1:])
        eqw_direct = zeros(shape(self.em_model)[1:])
        fit_func = lambda x, a ,b, c: a*exp(-(x-b)**2/2./c**2)

        for i,j in xy:
            cond = (self.fitwl > self.em_model[center_index,i,j]\
                - sigma_limit*self.em_model[sigma_index,i,j])\
                & (self.fitwl < self.em_model[center_index,i,j]\
                + sigma_limit*self.em_model[sigma_index,i,j])
            fit = fit_func(self.fitwl[cond],
                *self.em_model[[amp_index, center_index, sigma_index], i, j])
            cont = self.fitcont[cond,i,j]
            eqw_model[i,j] = trapz(1. - (fit+cont)/cont, x=self.fitwl[cond])
            eqw_direct[i,j] = trapz(1. - self.data[cond,i,j]/cont,
                x=self.restwl[cond])

        return array([eqw_model,eqw_direct])

    def plotfit(self, x, y):
        """
        Plots the spectrum and features just fitted.

        Parameters
        ----------
        x : number
            Horizontal coordinate of the desired spaxel.
        y : number
            Vertical coordinate of the desired spaxel.

        Returns
        -------
        Nothing.
        """

        fig = plt.figure(1)
        plt.clf()
        ax = plt.axes()

        p = self.em_model[:-1,y,x]
        c = self.fitcont[:,y,x]
        wl = self.fitwl
        f = self.fit_func
        s = self.fitspec[:,y,x]

        ax.plot(wl, c + f(wl, p))
        ax.plot(wl, c)
        ax.plot(wl, s)

        if self.fit_func == lprof.gauss:
            npars = 3
            parnames = ('A', 'wl', 's')
        elif self.fit_func == lprof.gausshermite:
            npars = 5
            parnames = ('A', 'wl', 's', 'h3', 'h4')
        else:
            raise NameError('Unkown fit function.')

        if len(p) > npars:
            for i in arange(0, len(p), npars):
                ax.plot(wl, c + f(wl, p[i:i+npars]), 'k--')

        pars = (npars*'{:10s}'+'\n').format(*parnames)
        for i in arange(0, len(p), npars):
            pars += (('{:10.2e}'+(npars-1)*'{:10.2f}'+'\n')\
                .format(*p[i:i+npars]))

        print pars
        plt.show()

    def channelmaps(self, channels=6, lambda0=None, velmin=None, velmax=None,
        continuum_width=300, continuum_opts=None, sigma=1e-16):
        """
        Creates velocity channel maps from a data cube.

        Parameters
        ----------
            channels : integer
                Number of channel maps to build
            lambda0 : number
                Central wavelength of the desired spectral feature
            vmin : number
                Mininum velocity in kilometers per second
            vmax : number
                Maximum velocity in kilometers per second
            continuum_width : number
                Width in wavelength for the continuum evaluation window
            continuum_opts : dictionary
                Dicitionary of options to be passed to the
                spectools.continuum function
        
        Returns
        -------
        """
        # Converting from velocities to wavelength
        wlmin, wlmax = lambda0*(array([velmin, velmax])/2.99792e+5 + 1.)
        wlstep = (wlmax - wlmin)/channels
        wl_limits = arange(wlmin, wlmax + wlstep, wlstep)


        side = int(ceil(sqrt(channels)))  # columns
        otherside = int(ceil(channels/side))  # lines
        fig = plt.figure()
        plt.clf()

        if continuum_opts == None:
            continuum_opts = {'niterate' : 3, 'degr' : 5,
                'upper_threshold' : 3, 'lower_threshold' : 3}
        cp = continuum_opts
        cw = continuum_width
        fw = lambda0 + array([-cw/2., cw/2.])

        cont = self.continuum(niterate=cp['niterate'],
            degr=cp['degr'], upper_threshold=cp['upper_threshold'],
            lower_threshold=cp['lower_threshold'],
            fitting_window=fw)
        contwl = self.wl[ (self.wl > fw[0]) & (self.wl < fw[1]) ]
        cont_wl2pix = interp1d(contwl, arange(len(contwl)))

        for i in arange(channels):
            ax = fig.add_subplot(otherside, side, i+1)
            wl = self.restwl
            wl0, wl1 = wl_limits[i], wl_limits[i+1]
            print wl[(wl > wl0) & (wl < wl1)]            
            wlc, wlwidth = average([wl0, wl1]), (wl1-wl0)

            f = self.wlprojection(wlc, fwhm=wlwidth, writefits=False,
                filtertype='box') - cont[int(round(cont_wl2pix(wlc)))]
            f[f < sigma] = nan
            cp = continuum_opts

            ax.imshow(f, interpolation='none', aspect=1)
            ax.annotate('{:.0f}'.format((wlc - lambda0)/lambda0*2.99792e+5),
                xy=(0.1, 0.8), xycoords='axes fraction', color='k')
            if i%side != 0:
                ax.set_yticklabels([])
            if i/float( (otherside-1)*side ) < 1:
                ax.set_xticklabels([])

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()
  
    def voronoi_binning(self, targetsnr=10.0, writefits=False,
                        outfile=None, clobber=True, writevortab=True):
        """
        Applies Voronoi binning to the data cube, using Cappellari's
        Python implementation.
    
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
        clobber : boolean
            Overwrites files with the same name given in 'outfile'.
        writevortab : boolean
            Saves an ASCII table with the binning recipe.
    
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
