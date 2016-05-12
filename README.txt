# ifscube

A set of python scripts and functions to analyse and process integral
field spectroscopy data cubes.

Author: Daniel Ruschel Dutra

Website: https://git.cta.if.ufrgs.br/ruschel/ifscube

## Installation instructions

We recommend installing the package via pip (https://pypi.python.org/pypi/pip),
which is bundled with the standard Python distribution since Python 3.4.

To install IFSCUBE using pip just enter the following in a terminal:

pip install git+https://git.cta.if.ufrgs.br/ruschel/ifscube.git


## Voronoi binning method

Many of the functions in 'ifscube' have provisions for working with data
binned via the Voronoi method. If you are not interested in binning your
data cubes, than you can simply comment the import line that says:


        from voronoi_2d_binning import voronoi_2d_binning


After that everything should work in a pixel by pixel basis.
However, if you want to process your data with Voronoi binning you
should download Michele Cappellari's python implementation from the
following address:

http://www-astro.physics.ox.ac.uk/~mxc/software/
