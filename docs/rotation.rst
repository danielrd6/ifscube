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

This command assumes that you have already run the cube fitting example with
the data for NGC3081.

Configuration file
------------------

The parameters in the configuration file are directly translated in to arguments
for the methods of the :class:`ifscube.rotation.Rotation` class.

general
*******

* fit: boolean
    If *yes* fits the data, if *no* just builds a model with the parameters given
    in the section :ref:`model`.


model
*****

bla