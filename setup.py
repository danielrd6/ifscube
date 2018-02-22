from numpy.distutils.core import Extension, setup


with open('./ifscube/.version', 'r') as verfile:
    __version__ = verfile.read().strip('\n')

ext1 = Extension(name='ifscube.elprofile',
                 sources=['ifscube/profiles.f90'])

packdata = {}
packdata['ifscube'] = [
    'examples/*',
    'examples/halpha.cfg',
    'examples/halpha_cube.cfg',
    'data/*',
    'docs/*']

setup(
    name='ifscube',
    version=__version__,
    packages=['ifscube'],
    package_data=packdata,
    scripts=['bin/fit_scrutinizer', 'bin/cubefit', 'bin/specfit'],
    description="Fit emssision lines",
    author="Daniel Ruschel Dutra",
    author_email="druscheld@gmail.com",
    url='https://git.cta.if.ufrgs.br/ruschel/ifscube',
    platform='Linux',
    license='GPLv3',
    ext_modules=[ext1],
    classifiers=[
      'Programming Language :: Python :: 3.5'
    ],
    install_requires=[
        'numpy>=1.12.0',
        'scipy>=0.18.1',
        'matplotlib>=1.5',
        'astropy>=1.5',
        'progressbar33>=1.0',
    ],
)
