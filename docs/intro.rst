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

Using Anaconda
------------------------

If you are familiar with Anaconda [#anaconda]_  and contained environments, I recommend you install IFSCube within its
own environment, in order to avoid conflicts with other packages and package versions.
The repository has an environment file named `environment.yml` which can be used to create a Conda environment with all
the requirements already installed.
The following command will create an environment named `ifscube` in your system, and subsequently switch to this
environment.

.. code-block:: bash

    git clone https://github.com/danielrd6/ifscube.git
    cd ifscube
    conda env create -f environmnent.yml
    conda activate ifscube

Once you are within the correct conda environment, you can install IFSCube with pip.
While still in the `ifscube` directory enter the following command:

.. code-block:: bash

    pip install .

Without Anaconda
--------------------------------------------------

IFSCube requires that some Fortran compiler be present in the system, and it is recommended to have Git_.
If you are using Ubuntu you can install them by using the following command:

.. code-block:: bash

    sudo apt update && sudo apt install gfortran git

In order to get the code you can use git and clone the entire repository.

.. code-block:: bash

    git clone https://github.com/danielrd6/ifscube.git

After that you will need to install the required python packages.
Switch to the directory where you cloned the repository (the default is ifscube), and run pip.

.. code-block:: bash

    cd ifscube
    pip install -r requirements.txt

Finally, while still in the `ifscube` directory, install the package with

.. code-block:: bash

    pip install .

If you are having trouble with the Fortran compiler you can force one with pip's install options, which are
exemplified below.

To force a specific compiler:

.. code-block:: bash

    pip install --install-option=build --install-option='--fcompiler=gnu95' .

Developer installation
------------------------

If you want to be able to change the package to suit your needs, or contribute
with your own code to the project, it is recommended to clone the git
repository and install the package as an editable package.

.. code-block:: bash

    pip install --editable .

Upgrade
--------------------------------------------------

If you want to upgrade an existing installation of IFSCube use

.. code-block:: bash

    pip install --upgrade git+https://github.com/danielrd6/ifscube.git

.. rubric:: Footnotes

.. [#astroconda] https://astroconda.readthedocs.io/en/latest/

.. [#anaconda] https://www.anaconda.com/

.. _git: https://git-scm.com/
