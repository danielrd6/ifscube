Welcome to ifscube's documentation!
===================================

Ifscube documentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   quickstart
   oned_fit
   cube_fit
   check_fit
   modules

What is IFSCube?
----------------

IFSCube is a python package designed to perform analysis tasks in data
cubes of integral field spectroscopy. It was originally designed to work
with the data from Gemini’s Multi-Object Spectrograph (GMOS), and as
such, many default parameters are set to work best with this kind of
data. Using it with data cubes from other instruments should be possible
by explicitly setting a few more parameters.

Keep in mind that this is not a closed software that will be called from
the command line and perform the tasks that you might want. Instead it
is intended to be used as part of your own scripts, hopefully making
them a lot more concise, and saving you a lot of time. At the same time,
since everything you can do in a python script can also be done using an
interactive interpreter such as ipython, you can also perform your
analysis on the fly.

For the specific task of fitting spectral features there is an
executable script that can be called from the command line, and
controlled via an ASCII configuration file. This is the recommended
method for line fitting, as many of the preparation steps can be easily
set up by the configuration parser.

Installation instructions
-------------------------

The preferred method for the installation of IFSCube is to use pip. Pip
can install directly from the git repository using the following
command:

::

    pip install git+https://danielrd6@bitbucket.org/danielrd6/ifscube.git

IFSCube uses third party programs that are not distributed with the
package, namely pPXF and Voronoi Binning, written by Michelle
Cappellari. If you wish to use IFSCube in conjunction with these
programs you will have to install them first, by downloading and
following the instructions at the following address:
http://www-astro.physics.ox.ac.uk/mxc/software/

Upgrade
-------

If you want to upgrade an existing installation of IFSCube use

::

    pip install --upgrade git+https://danielrd6@bitbucket.org/danielrd6/ifscube.git

If you are having trouble with the Fortran compiler you can force one
with pip’s install options, which are exemplified below.

To force a specific compiler:

::

    pip install git+https://danielrd6@bitbucket.org/danielrd6/ifscube.git --install-option=build --install-option='--fcompiler=gnu95'

IFSCube has been extensively tested with the astroconda [1]_
distribution, therefore it is highly recommended that you install it
within the astroconda’s Python 3 environment.

If you want to be able to change the package to suit your needs, or
contribute with your own code to the project, it is recommended to clone
the git repository and install the package as an editable package.

::

    git clone https://danielrd6@bitbucket.org/danielrd6/ifscube.git
    cd ifscube
    pip install -e .



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [1]
   https://astroconda.readthedocs.io/en/latest/
