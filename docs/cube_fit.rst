Datacube fitting
****************

Using IFSCUBE to fit emission lines in data cubes is very similar to
fitting a single spectrum, which is described in section
:doc:`oned_fit`. To start the fitting process you must call the
:program:`cubefit` executable script from the command line.

::

    cubefit -c halpha_cube.cfg ngc3081_cube.fits

Here we are using the provided example files distributed with ifscube.

For more information on the available command line options of :program:`cubefit`,
please read the help page printed by

::

    cubefit -h

Configuration file
====================

There are only minor differences between the configuration files of
:program:`specfit` and :program:`cubefit`, which will be covered in the following subsections,
each relating to a particular section of the configuration file. Please
refer to section :doc:`oned_fit` for parameters and options that
also apply to single spectrum fitting.

fit
---

* bounds_change: <change_par_0>, <change_par_1>, ...
    When using refit (see below), set the bounds to these distances from the
    nearby successful fits. The values are all aditive, with the exception
    of the amplitude (always the first parameter), which is a ratio.
    For instance, if you want the amplitude to change by at most half the
    current value, the velocity to change by at most 30 km/s and the
    velocity dispersion by 20 km/s, you would write "0.5, 30, 20".

* individual_spec: 'no', 'x, y', 'peak' or 'cofm'
    If set to *no* fits all the spectra in the data cube, else fits only
    one spectrum. If set to *x, y* fits the spectrum in the spaxel with
    horizontal coordinate *x* and vertical coordinate *y*. *peak* will
    fit only the spaxel with the highest value in an image resulting
    from the sum of all the pixels along the dispersion direction.
    *cofm* is similar to *peak*, but uses the center of mass instead.

* refit: 'yes', 'no'
    Uses parameters from previous successful fits as the initial guess
    for subsequent fits. The parameters are the average of the results
    for fits returning a fit\_status of 0 within a given refit\_radius.

* refit_radius: float
    Radius in pixels to use when averaging parameters for the updated
    initial guess.

* spiral_loop*: 'yes', 'no'
    Fits the spaxels following a spiral pattern from the specified
    spiral_center outwards. This is particularly useful when refit is
    set to ’yes’, since the algorithm will start from the highest
    signal to noise ratio spectra.

* spiral_center: 'x, y', 'peak' or 'cofm'
    Chooses where the spiral pattern will start. See *individual_spec*
    above for a description of the meaning of 'x, y', ’peak’ and
    'cofm'.
