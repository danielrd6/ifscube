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

If you are familiar with UV [#uv]_, Anaconda [#anaconda]_, or other 
contained environments, I recommend you install IFSCube within its
own environment, in order to avoid conflicts with other packages and 
package versions.

The following commands will copy IFSCube's git repository to your computer, and then move to the newly created directory containing the code. 

.. code-block:: bash

    git clone https://github.com/danielrd6/ifscube.git
    cd ifscube

From here you can install it directly with `pip`.
If you are using an environment manager, remember to change to the appropriate environment before using `pip`.

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

issued from within the `ifscube` directory, and avoid the need to run `pip 
install` every time there is an update.
Additionally, you can checkout different branches with

.. code-block:: bash

    git checkout <branch_name>

Experimental features are always first available in dedicated branches before being incorporated into the main version.

Alternatively, one can also install `ifscube` directly from the git repository with

.. code-block:: bash

    pip install git+https://github.com/danielrd6/ifscube.git

Similarly, upgrades can also be done directly from the online repository with

.. code-block:: bash

    pip install --upgrade git+https://github.com/danielrd6/ifscube.git

.. rubric:: Footnotes

.. [#astroconda] https://astroconda.readthedocs.io/en/latest/

.. [#anaconda] https://www.anaconda.com/

.. [#anaconda] https://docs.astral.sh/uv/

.. _git: https://git-scm.com/
