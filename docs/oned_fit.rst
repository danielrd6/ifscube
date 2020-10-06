Single spectrum fitting
************************************************************

One of the functions available for IFSCube allows for the fitting of a single
1D spectrum. This is very useful when experimenting parameters for the
fits using a combined, higher signal-to-noise, spectrum from the data cube.
There is even a executable script to perform the fits using parameters given in
a configuration file.

:program:`specfit` usage
============================================================

After the installation of IFSCube the executable :program:`specfit` should be
available in your path. If not, the script is located in the bin directory of
the IFSCube installation directory.

The simplest way to use the program is to just invoke it as

.. code-block:: bash

    specfit -c halpha.cfg manga_onedspec.fits

Here, and in all subsequent examples, we will use the data available
in the ifscube/examples directory. The above command specifies the
configuration file with the **-c** option. If you want to know more about the
command line options of linefit just execute it with the **-h** option, and a
help page will be printed.

The configuration file halpha.cfg, present in the examples directory, showcases
the syntax and some of the possibilities of :program:`specfit`. The reader is
strongly encouraged to start with that file and modify it for her/his fits.

Configuration file
============================================================

The configuration file for :func:`specfit` follows the formalism of typical
*.ini* files, with sections defined by strings within brackets and parameter as
strings followed by *:* or *=* and the corresponding value. Comments are
possible, and are declared by either a **#** or a **;**. In the following
subsections each section of the configuration file will be discussed in more
detail. Boolean options, such as fit\_continuum and overwrite take 'yes' or
'no' as values.


fit
---

This part of the configuration file sets the main options of the fitting
process. 

* continuum_line_weight: number
    When fitting a pseudo-continuum function, sets the weight of pixels near
    spectral features to this number. Setting this option to 1.0 effectively
    disables it, while setting it to zero makes the pseudo-continuum fitting
    completely insensitive to any pixels within 3 times the given sigma from
    the line center.
* fit_continuum: 'yes', 'no'
    Fits a polynomial pseudo continuum before fitting the spectral features.
* fitting_window: lambda_0:lambda_1
    Spectral window in which to perform the fit.
* function: 'gaussian', 'gauss_hermite'
    Sets the function to be used as the spectral feature profile. It can be
    either 'gaussian' or 'gauss_hermite'.
* guess_parameters: 'yes', 'no'
    Makes an initial guess for the amplitude, centroid and sigma of each
    spectral feature based on the spectrum. Setting this option to yes
    *does not* mean that you can leave the line definition sections empty. A lot
    of other routines within the algorithm are based on the initial parameters you
    give for each spectral feature.
* method: any in scipy.optimize.minimize, 'differential_evolution'
    Method of minimization. If you are really unsure about the parameters for your fit, I recommend using
    differential_evolution, since it is immune to local minima.
* monte_carlo: integer
    Number of Monte Carlo iterations for uncertainty estimates. The uncertainties will only be meaningful
    if the given variance is correct.
* optimize_fit: 'yes', 'no'
    Only fits pixels that are close to the spectral features set in the
    configuration file. For instance, if you want to fit a spectrum that goes from
    4800A to 7000A, but you are only interested in the [O III] 5007 and [N II] 6583
    lines, you can set this option to 'yes', and save the computing time required
    for all the zeros in between.
* optimization_window: number
    Size of the optimized fitting window in units of the sigma given as initial
    guess. If optimize_fit is set to yes (see above) Only pixels with wavelength
    between (wavelength - optimization_window * sigma) and (wavelength +
    optimization_window * sigma) will be evaluated by the fitting algorithm.
* out_image: string
    Name of the output FITS file.
* overwrite: 'yes', 'no'
    Overwrites the output file if it already exists.
* suffix: string
    Suffix to attach to the name of the input file. The resulting concatenation
    will be the output file's name.
* verbose: 'yes', 'no'
    Shows a nice progress bar.
* write_fits: 'yes', 'no'
    Writes the output of the fit to a file.  


loading
-------

The **loading** section is dedicated to parameters that tell :program:`specfit` how
to load your spectrum from the FITS file. Each parameter listed below
takes as input value a string that should match the name of the FITS
extension in the input MEF file containing the appropriate data. It is
important to point out that all the extensions must match the dimensions
of the observed spectrum, except for the primary, which should only
contain a header.

* scidata:
    Scientific data, or the actual observed spectrum.

* primary:
    Primary extension, with the main header.

* variance:
    Pixel by pixel variance.

* stellar:
    Stellar spectrum to be subtracted from the observed
    spectrum before the fit.

* flags:
    Flag spectrum, with zeros setting value that should not be
    used.

* redshift:
    This is the only parameter that is not supposed to be a FITS extension.
    specfit is designed to read a redshift from the primary extension header.
    If a ’redshift’ keyword is not found, it tries to read the redshift given
    in the configuration file. If none is given in either way, the spectrum is
    assumed be to already in the rest frame.

minimization
------------

This section controls the minimization algorithm, and its parameters are
directly passed on to the *scipy.optimize.minimize* function. A number
of different solvers are accessible via the *minimize* function, but
currently specfit only The reader is encouraged to read the
documentation for the scipy function in order to gain a deeper
understanding of the fitting process. In the parameter list below a few
example values are offered as a suggestion.

* eps: (1e-2) number
    Step size used for numerical approximation of the jacobian.

* ftol: (1e-5) number
    Precision goal for the value of f in the stopping criterion.

* disp: ’yes’, ’no’
    Displays detailed information of the fit.

* maxiter: 100 number
    Maximum number of minimization iterations.

continuum
---------

This part of the configuration file sets the parameters for the fitting
of the pseudo continuum. The continuum is defined as a Legendre polynomial of
arbitrary degree, which is fit to the spectrum after the subtraction of
the stellar component, if there is one.

Emission lines and other data points that should not be considered in
the continuum fit are eliminated via an iterative rejection algorithm.
For this reason, the fitting\_window set in the *fit* section should
provide enough room for an adequate sampling of valid continuum points.

* degree: integer
   Degree of the polynomial.

* n_iterate: integer number
   Number of rejection iterations.

* lower / upper\_threshold: number
   The rejection threshold in units of standard deviation.

Feature definition
==================

Features to be fitted are defined as sections with arbitrary names, as long
as these names are not *fit*, *continuum*, *minimization* and *loading*,
which are reserved.
The basic syntax for a feature, or spectral line, definition is as
follows:

::

    [feature_name]
    <paremeter0>: <value>, <bounds>, <constraints>
    <paremeter1>: <value>, <bounds>, <constraints>
    ...

Parameters
----------

The valid parameters are for each feature are: rest_wavelength, velocity
sigma, amplitude, k_group and continuum_windows. With the exception of
**rest_wavelength**, **k_group** and **continuum_windows**, all the
values for each parameter are in fact initial guesses for the fitter, unless
they are explicitly defined as fixed values.
We will now discuss each these in more detail:

* rest_wavelength:
    The wavelength of the spectral feature (or line) to be fit as it
    is observed in the rest frame. The accuracy of this parameter is
    very important, as all the velocity evaluations will be based on this value.
    Units for these parameter are the same as the input spectrum.

* velocity:
    Centroid velocity of the spectral feature in units of km/s. Blue shifted
    lines have negative velocity, while red shifted ones have positive velocity.

* sigma:
    The second moment of the Gaussian or Gauss-Hermite polynomial, commonly
    known as the standard deviation. It should be given in units of km/s.

* amplitude:
    Amplitude of the Gaussian function or Gauss-Hermite polynomial in units
    of the input spectrum.

* fixed:
    If set to yes, this spectral feature will have all its parameters fixed, except for the amplitude, which will be
    fixed relative to all the other features that are set as fixed. This is specially useful for fitting broad emission
    lines in Seyfert 1's.

All the above parameters are mandatory for every spectral feature, except if
they are part of a kinematic group.
The last two parameters that a spectral feature can take are optional,
and deserve a somewhat more detailed explanation.

Lastly, **continuum_windows** specifies the windows for the pseudo
continuum fitting used in the equivalent width evaluation, and are not
used anywhere else. It should be given as four wavelength values
separated by commas.


Kinematic grouping
------------------

Many spectral features are physically linked, being produced in the same regions of the astronomical target.
The parameter **k_group** stands for kinematic grouping, and it is an automated way to specify that
the Doppler shift and velocity dispersion of all features sharing the same **k_group** should be equal.
In order to use this parameter, you only need to specify an arbitrary integer number as the
value for a given feature, and repeat that same number for all other features sharing the same kinematics.

Internally IFSCube ignores the kinematic parameters of all features in the same group, except for the first, which is
the only one passed on to the minimization algorithm.
For each residual evaluation, the model spectra is generated by copying the same exact kinematic parameters for every
line in a **k_group**.
As a result of this implementation, it is irrelevant to add kinematic parameters to more than one feature within the
same **k_group**.
The best practice is to define all the parameters for the first feature, and use only **rest_wavelength**,
**amplitude**, **k_group** and **continuum_windows** for the others.

This method differs from traditional non-linear constraining of parameters, specially because it reduces the number of
parameters and functions being evaluated.
For instance, if you are trying to fit Gaussian curves to three emission lines, and they all share the same kinematics,
the number of parameters passed to the minimization algorithm will be five: three amplitudes, one velocity and one
velocity dispersion.


Bounds
------

Bounds for each parameter are given in one of two ways: i) two values
separated by a **:**, or ii) a single value preceded by **+-**. For
instance, if you want to set the wavelength for a given feature

::

    velocity: 300.0, 1000:500.0

or

::

    velocity: 300.0, +- 200

**do not forget** the space between **+-** and the number that follows it.

Bounds can also be one-sided, as in

::

    amplitude: 1e-15, 1e-19:

which will be interpreted as having only the lower limit of 1e-19 and no
upper limit.

Constraints
-----------

Constraints are perhaps the most valuable tool for any spectral feature
fitting. We already discussed the automated constraints that keep the
same kinematic parameters for different spectral features using the
**k_group** parameter, but :mod:`specfit` also accepts arbitrary relations
between the parameters of different features. For instance, suppose
you want fix the flux relation between two lines you know to be
physically connected, such as the [N II] lines at 6548A and 6583A.

::

    [n2_a]
    rest_wavelength: 6548
    velocity: 0
    sigma: 60
    amplitude: 1e-15,, n2_b.amplitude / 3
    k_group: 0

    [n2_b]
    rest_wavelength: 6583
    velocity: 0
    sigma: 60
    amplitude: 1e-15
    k_group: 0

The double comma before the constraint is there because value, bounds
and constraints are separated by commas, and even if you do not want to
set any bounds, an extra comma is necessary for the parser to correctly
identify the constraint.

Now let us discuss the syntax of the constraint, which is the expression
**n2_b.amplitude / 3**.
The parser accepts simple arithmetic operations (\*, /,
+, -), inequality relations (:math:`<`, :math:`>`), numbers and feature
parameters.
Feature parameters are given as **<feature_name>.<parameter_name>**.
For that reason feature names **should not** include periods.
