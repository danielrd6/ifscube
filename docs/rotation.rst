Fit rotation model
==================

Here is a little introduction to the process of fitting a rotation model
to a velocity field. IFSCube has a program for that called :program:`fit_rotation`,
which takes a velocity field from a FITS file and attempts to fit the disk
rotation model described in Bertola 1991.

In the examples directory there is a configuration file for :program:`fit_rotation`.
After installation the program should be available at your path, so that you
can call it by entering the following in a terminal, from within the examples
directory.

.. code-block:: bash

    fit_rotation rotation.ini

This command assumes that you have already run the :doc:`cube fitting <cube_fit>`
example with the data for NGC3081.

Configuration file
------------------

The parameters in the configuration file are directly translated in to arguments
for the methods of the :class:`ifscube.rotation.Rotation` class.

general
*******

* fit: bool
    If *yes* fits the data, if *no* just builds a model with the parameters given
    in the section :ref:`model`.


loading
*******

* input_data: str
    Name of the input FITS file containing the velocity field data.
* extension: str or int
    Name or number of the extension in case of a multi-extension FITS file.
* plane: int
    Plane of the velocity field, in case of data cube.

.. _model:

model
*****

The model parameters are equal to those of Bertola et al. 1991, except for *x_0* and *y_0*.
All angles should be given in degrees.

* amplitude: float
    Amplitude of the velocity field.
* x_0: float
    Horizontal coordinate of the rotation center.
* y_0: float
    Vertical coordinate of the rotation center.
* phi_0: float
    Angle of the line of nodes.
* c_0: float
    Concentration index.
* p: float
    *p* parameter.
* theta: float
    Inclination angle of the disk with respect to the plane of the sky.

bounds
******

For each parameter of the last section a pair of bounds can be set in this section,
repeating the parameter name and given lower and upper bounds separated by a comma.

.. code-block:: ini

    [bounds]
    theta = 30, 60

fixed
*****

Parameters can also be set to be fixed, by writing the parameter name in this section
followed by a 'yes'.

.. code-block:: ini

    [fixed]
    x_0 = 'yes'
    y_0 = 'yes'