Introduction
********************

What is IFSCube?
====================

IFSCube is a python package designed to perform analysis tasks in data cubes of integral field spectroscopy. It was originally intended
to be used as part of larger scripts, hopefully making them a lot more
concise, and saving a lot of time. However, since everything you
can do in a Python script can also be done using an interactive interpreter
such as ipython, you can choose to perform your analysis on the fly.

During development I found out that some tasks work better as executable
programs that the user could call directly from the command line, with
options being set by an ASCII configuration file. These tasks are the 
fitting of spectral features in 1D spectra and data cubes, the fitting
of rotation models to velocity fields and the inspection of fit results,
which are accomplished by the programs :program:`specfit`, :program:
`cubefit`, :program:`fit_rotation` and :program:`fit_scrutinizer`
respectively.

The preferred data format for IFSCube is the Flexible Image Transport 
System (FITS) standard. However, if you wish to use other formats or 
perform some preliminary processing of your data, it is enough to subclass 
the :class:`ifscube.datacube.Cube` or :class:`ifscube.onedspec.Spectrum` 
classes, substituting the :meth:`__init__` and :meth:`_load` methods.

Acknowledging IFSCube
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

IFSCube is a pure Python package, so it needs no compiler. The recommended
way to install it is with ``pip`` (or ``uv``) inside a virtual environment.
Conda is no longer required.

Installing from git
~~~~~~~~~~~~~~~~~~~

Create and activate a virtual environment, then install directly from the
repository:

.. code-block:: bash

    python -m venv ifscube-env
    source ifscube-env/bin/activate
    pip install git+https://github.com/danielrd6/ifscube.git

If you use `uv <https://docs.astral.sh/uv/>`_, the equivalent is:

.. code-block:: bash

    uv venv
    uv pip install git+https://github.com/danielrd6/ifscube.git

To upgrade an existing installation:

.. code-block:: bash

    pip install --upgrade git+https://github.com/danielrd6/ifscube.git

Installing for development
~~~~~~~~~~~~~~~~~~~~~~~~~~

To work on the code, clone the repository and install it in editable mode,
so that changes take effect without reinstalling:

.. code-block:: bash

    git clone https://github.com/danielrd6/ifscube.git
    cd ifscube
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .

With uv, ``uv sync`` creates the environment and installs the project and
its dependencies in one step.

You can check out other branches with ``git checkout <branch_name>``.

Third party software
~~~~~~~~~~~~~~~~~~~~

IFSCube uses third party programs that are not distributed with the package,
namely `pPXF <https://pypi.org/project/ppxf/>`_ and `PowerBin <https://pypi.org/project/powerbin/>`_.
See their documentation for installation instructions.

Free-threaded Python (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IFSCube can be installed on free-threaded builds of Python (for example
3.14t), but support is experimental. This requires numba 0.65 or later,
which is the first release to support free-threaded Python.

At the time of writing, h5py does not publish free-threaded wheels, so
pip and uv will try to build it from source. This needs the HDF5 libraries
and ``pkg-config``. Check whether newer h5py releases provide ``cp314t``
wheels first, in which case none of the steps below are needed.

On macOS (with Homebrew):

.. code-block:: bash

    brew install hdf5 pkg-config
    export HDF5_DIR="$(brew --prefix hdf5)"

On Debian or Ubuntu:

.. code-block:: bash

    sudo apt install libhdf5-dev pkg-config

Then install IFSCube as usual. To confirm that the GIL is actually disabled
after importing it, run:

.. code-block:: bash

    python -c "import ifscube, sys; print(sys._is_gil_enabled())"

This should print ``False``. If it prints ``True``, one of the compiled
dependencies has re-enabled the GIL.

.. note::

    h5py serializes its own calls with a global lock, even without the GIL,
    so reading data files will not speed up with threads. Parallel speedups
    come from the numerical code.
