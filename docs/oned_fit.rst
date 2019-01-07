Single spectrum fitting
************************************************************

One of the functions available for IFSCube allows for the fitting of a single
1D spectrum. This is very useful when experimenting parameters for the
fits using a combined, higher signal-to-noise, spectrum from the data cube.
There is even a executable script to perform the fits using parameters given in
a configuration file.

:func:`specfit` usage
============================================================

After the installation of IFSCube the executable {\sc specfit} should be
available in your path. If not, the script is located in the bin directory of
the IFSCube installation directory.

The simplest way to use the program is to just invoke it as

.. code-block:: bash

    specfit -c halpha.cfg manga_onedspec.fits

Here, and in all subsequent examples, we will use the data available
in the ifscube/examples directory. The above command specifies the
configuration file with the {\bf -c} option. If you want to know more about the
command line options of linefit just execute it with the {\bf -h} option, and a
help page will be printed.

The configuration file halpha.cfg, present in the examples directory, showcases
the syntax and some of the possibilities of {\sc specfit}. The reader is
strongly encouraged to start with that file and modify it for her/his fits.

Configuration file
============================================================

The configuration file for *specfit* follows the formalism of typical {\it
.ini} files, with sections defined by strings within brackets and parameter as
strings followed by *:* or *=* and the corresponding value. Comments
are possible, and are declared by either a {\bf \#} or a {\bf ;}. In the
following subsections each section of the configuration file will be discussed
in more detail. Boolean options, such as fit\_continuum and overwrite take 'yes' or
'no' as values.


fit
---

This part of the configuration file sets the main options of the fitting
process. 

* fit_continuum: 'yes', 'no'
    Fits a polynomial pseudo continuum before fitting the spectral features.

* function: 'gaussian', 'gausshermite'
    Sets the function to be used as the spectral feature profile. It can be
    either 'gaussian' or 'gausshermite'. 
* fitting_window: lambda_0:lambda_1
    Spectral window in which to perform the fit.
* outimage: string
    Name of the output FITS file.
* overwrite: 'yes', 'no'
    Overwrites the output file if it already exists.
* optimize_fit: 'yes', 'no'
    Only fits pixels that are close to the spectral features set in the
    configuration file. For instance, if you want to fit a spectrum that goes from
    4800A to 7000A, but is only interested in the [O III] 5007 and [N II] 6583
    lines, you can set this option to 'yes', and save the computing time required
    for all the zeros in between.
* optimization_window: number
    Size of the optimized fitting window in units of the sigma given as initial
    guess. If optimize_fit is set to yes (see above) Only pixels with wavelength
    between (wavelength - optimization_window * sigma) and (wavelength +
    optimization_window * sigma) will be evaluated by the fitting algorithm.
* suffix: string
    Suffix to attach to the name of the input file. The resulting concatenation
    will be the output file's name.
* verbose: 'yes', 'no'
    Shows a nice progress bar.
* writefits: 'yes', 'no'
    Writes the output of the fit to a file.  
* guess_parameters: 'yes', 'no'
    Makes an initial guess for the amplitude, centroid and sigma of each
    spectral feature based on the spectrum. Setting this option to yes
    *does not* mean that you can leave the line definition sections empty. A lot
    of other routines within the algorithm are based on the initial parameters you
    give for each spectral feature.
* test_jacobian: 'yes', 'no'
    Checks if there are null values in the jacobian matrix of the fit. If there
    are, it usually means that the spectral feature is in a flagged section of
    the spectrum, or that the best fit is a line with zero amplitude.

loading
-------

The *loading* section is dedicated to parameter that tell {\sc specfit} how to
load your spectrum from the FITS file. Each parameter listed below takes as
input value a string that should match the name of the FITS extension in the
input MEF file containing the appropriate data. It is important to point out
that all the extensions must match the dimensions of the observed spectrum,
except for the primary, which should only contain a header. 

\begin{itemize}
  \item {\bf scidata}: Scientific data, or the actual observed spectrum.
  \item {\bf primary}: Primary extension, with the main header.
  \item {\bf variance}: Pixel by pixel variance.
  \item {\bf stellar}: Stellar spectrum to be subtracted from the observed
    spectrum before the fit.
  \item {\bf flags}: Flag spectrum, with zeros setting value that should not be
    used.
  \item {\bf redshift}: This is the only parameter that is not supposed to be
    a FITS extension. {\sc specfit} is designed to read a redshift from the
    primary extension header. If a 'redshift' keyword is not found, it tries to
    read
    the redshift given in the configuration file. If none is given in either
    way, the spectrum is assumed be to already in the rest frame.
\end{itemize}

\subsection{minimization}

This section controls the minimization algorithm, and its parameters are
directly passed on to the {\it scipy.\-optimize.\-minimize} function. A number
of different solvers are accessible via the {\it minimize} function, but
currently {\sc specfit} only 
The
reader is encouraged to read the documentation for the scipy function in order
to gain a deeper understanding of the fitting process. In the parameter list
below a few example values are offered as a suggestion.

\begin{itemize}
  \item {\bf eps}: (1e-2) number \\ Step size used for numerical approximation
    of the jacobian.
  \item {\bf ftol}: (1e-5) number \\ Precision goal for the value of f in the
    stopping criterion.
  \item {\bf disp}: 'yes', 'no' \\ Displays detailed information of the fit.
  \item {\bf maxiter}: 100 number \\ Maximum number of minimization iterations.
\end{itemize}

\subsection{continuum}

This part of the configuration file sets the parameters for the fitting of the
pseudo continuum. The continuum is defined as a polynomial of arbitrary degree,
which is fit to the spectrum after the subtraction of the stellar component, if
there is one.

Emission lines and other data points that should not be considered in the
continuum fit are eliminated via an iterative rejection algorithm. For this
reason, the fitting\_window set in the {\it fit} section should provide enough
room for an adequate sampling of valid continuum points.

\begin{itemize}
  \item {\bf degr}: integer number \\ Degree of the polynomial.
  \item {\bf niterate}: integer number \\ Number of rejection iterations.
  \item {\bf lower / upper\_threshold}: number \\ The rejection threshold in
    units of standard deviation.
\end{itemize}


\section{Feature definition}

Features to be fitted are defined as sections with arbitrary names, with the
exception of fit, minimization and continuum, which are reserved. The basic
syntax for a feature, or spectral line, definition is as follows:

\begin{verbatim}
[feature_name]
<paremeter0>: <value>, <bounds>, <constraints>
<paremeter1>: <value>, <bounds>, <constraints>
...
\end{verbatim}

\subsection{Parameters}

The valid parameters are for each feature are: wavelength, sigma, flux,
k\_group and continuum\_windows. Wavelength, sigma and flux are mandatory for
every spectral feature, and are pretty much self explanatory. Note that here
{\bf sigma is given in units of wavelength}. The last two parameters are
optional, and deserve some explanation. 

The parameter {\bf k\_group} stands for kinematic grouping, and it basically is
an automated way to specify that the Doppler shift and sigma of all features
sharing the same {\bf k\_group} should be equal. To set it, one only needs to
specify an arbitrary integer number as the value for a given feature, and
repeat that same number for all other features sharing the same kinematics.

Lastly, {\bf continuum\_windows} specifies the windows for the pseudo continuum
fitting used in the equivalent width evaluation, and are not used anywhere
else. It should be given as four wavelength values separated by commas.

\subsection{Bounds}

Bounds for each parameter are given in one of two ways: i) two values separated
by a {\bf :}, or ii) a single value preceded by {\bf +-}. For instance, if you
want to set the wavelength for a given feature

\begin{verbatim}
wavelength: 6562.8, 6552.8:6572.8
\end{verbatim}

or

\begin{verbatim}
wavelength: 6562.8, +-10 
\end{verbatim}

Bounds can also be one-sided, as in

\begin{verbatim}
flux: 1e-15, 1e-19:
\end{verbatim}

\noindent which will be interpreted as having only the lower limit of 1e-19 and
no upper limit.

\subsection{Constraints}

Constraints are perhaps the most valuable tool for any spectral feature
fitting. We already discussed the automated constraints that keep the same
kinematical parameters for different spectral features using the {\bf k\_group}
parameter, but {\sc specfit} also accepts arbitrary relations between the same
parameter of different features. For instance, suppose you want fix the flux
relation between two lines you know to be physically connected, such as the
[N {\sc ii}] lines at 6548\AA and 6583\AA.

\begin{verbatim}
[n2_a]
wavelength: 6548
sigma: 2
flux: 1e-15,, n2_b / 3
k_group: 0

[n2_b]
wavelength: 6583
sigma: 2
flux: 1e-15
k_group: 0
\end{verbatim}

\noindent The double comma before the constraint is there because value, bounds
and constraints are separated by commas, and even if you do not want to set any
bounds, an extra comma is necessary for the parser to correctly identify the
constraint.

Now let us discuss the syntax of the constraint, which is the
expression {\bf n2\_b / 3}. The parser accepts simple arithmetic operations
(*, /, +, -), inequality relations ($<$, $>$), numbers and feature names. The
feature name is the name given to the section containing the spectral feature
parameters, and the parameters constrained are always the same parameters in
different features. Currently the parser does not support relating the sigma of
some line to the flux of some other line.


