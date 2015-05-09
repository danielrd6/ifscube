# ifscube
A set of python scripts and functions to analyse and process integral field spectroscopy data cubes.


 -- Voronoi binning method --

Many of the functions in 'ifscube' have provisions for working with data
binned via the Voronoi method. If you are not interested in binning your
data cubes, than you can simply comment the import line that says:

```python
from voronoi_2d_binning import voronoi_2d_binning
```

After that everything should work in a pixel by pixel basis.
However, if you want to process your data with Voronoi binning you
should download Michele Cappellari's python implementation from the
following address:

http://www-astro.physics.ox.ac.uk/~mxc/software/
