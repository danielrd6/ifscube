Introduction
********************

What is IFSCube?
====================

IFSCube is a python package designed to perform analysis tasks in data
cubes of integral field spectroscopy. It was originally intended
to be used as part of your own scripts, hopefully making them a lot more
concise, and saving you a lot of time. However, since everything you
can do in a Python script can also be done using an interactive interpreter
such as ipython, you can choose to perform your analysis on the fly.

During development I found out that some tasks work better as executable
programs that the user could call directly from the command line, with
options being set by an ASCII configuration file. These tasks are the fitting of
spectral features in 1D spectra and data cubes, the fitting of rotation models
to velocity fields and the inspection of fit results, which are accomplished
by the programs :program:`specfit`, :program:`cubefit`, :program:`fit_rotation`
and :program:`fit_scrutinizer` respectively.

The preferred data format for IFSCube is the Flexible Image Transport System (FITS) standard. However, if you wish to
use other formats or perform some preliminary processing of your data, it is enough to subclass the
:class:`ifscube.datacube.Cube` or :class:`ifscube.onedspec.Spectrum` classes, substituting the :meth:`__init__`
and :meth:`_load` methods.

Installation instructions
==================================================

The preferred method for the installation of IFSCube is to use pip. Pip
can install directly from the git repository using the following command:

.. code-block:: bash

    pip install git+https://github.com/danielrd6/ifscube.git

Requirements
--------------------------------------------------

Apart from the Python modules listed in the requirements.txt file,
IFSCube also requires that some Fortran compiler be present in the system.
If you are using Ubuntu you can install one using the following command:

.. code-block:: bash

    sudo apt-get update && sudo apt-get install gfortran

Upgrade
--------------------------------------------------

If you want to upgrade an existing installation of IFSCube use

.. code-block:: bash

    pip install --upgrade git+https://github.com/danielrd6/ifscube.git

If you are having trouble with the Fortran compiler you can force one with
pip's install options, which are exemplified below.

To force a specific compiler:

.. code-block:: bash

    pip install --install-option=build --install-option='--fcompiler=gnu95' git+https://github.com/danielrd6/ifscube.git

IFSCube has been extensively tested with the
astroconda [#astroconda]_ distribution.
If you are not familiar with [#anaconda_], and do not know how to work with
multiple environments, I recommended that you install it within the astroconda's
Python 3 environment.

If you want to be able to change the package to suit your needs, or contribute
with your own code to the project, it is recommended to clone the git
repository and install the package as an editable package.

.. code-block:: bash

    git clone https://github.com/danielrd6/ifscube.git
    cd ifscube
    pip install -e .

If you are using Conda you can substitute the last line by

.. code-block:: bash

    conda develop -b .

Remember to switch to the desired Conda environment prior to running this command.

.. rubric:: Footnotes

.. [#astroconda] https://astroconda.readthedocs.io/en/latest/

.. [#anaconda] https://www.anaconda.com/
