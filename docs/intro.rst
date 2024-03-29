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

Acknowleging IFSCube
====================

If you use IFSCube in your research, and feel that it has contributed
significantly to your work, please consider citing the following paper

`Ruschel-Dutra et al. 2021, <https://ui.adsabs.harvard.edu/abs/2021MNRAS.507...74R/abstract>`_

which has been the main driver for the development of the code,
and the Zenodo DOI

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4065550.svg
   :target: https://doi.org/10.5281/zenodo.4065550

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
If you want to give another name to the ifscube environment, all you have to do is change the first line in the
`environment.yml` file.

.. code-block:: bash

    git clone https://github.com/danielrd6/ifscube.git
    cd ifscube
    conda env create -f environment.yml
    conda activate ifscube

Once you are within the correct conda environment, you can install IFSCube with pip.
While still in the `ifscube` directory enter the following command:

.. code-block:: bash

    pip install .

If you want to be able to modify the code to better suit your needs, or to contribute
your own code to the project, or just to be able to check different branches of the
source code, I recommend using the editable flag of pip's install.
Just substitute the line in the previous block by

.. code-block:: bash

    pip install -e .

This editable installation will allow you to update the code with a simple

.. code-block:: bash

    git pull

issued from within the `ifscube` directory, and avoid the need to run `pip install` every time there is an update.
Additionally, you can checkout different branches with

.. code-block:: bash

    git checkout <branch_name>

Experimental features are always first available in dedicated branches before being incorporated into the main version.

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
