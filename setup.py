from numpy.distutils.core import Extension, setup


with open('./ifscube/.version', 'r') as verfile:
    __version__ = verfile.read().strip('\n')

ext1 = Extension(name='ifscube.elprofile',
                 sources=['ifscube/profiles.f90'])

packdata = {
    'ifscube': [
        'examples/*',
        'examples/halpha.cfg',
        'examples/halpha_cube.cfg',
        'data/*',
        'docs/*'
    ]
}

setup(
    name='ifscube',
    python_requires='>3.6',
    version=__version__,
    packages=['ifscube'],
    package_data=packdata,
    scripts=['bin/fit_scrutinizer', 'bin/cubefit', 'bin/specfit', 'bin/fit_rotation'],
    description="Fit emssision lines",
    author="Daniel Ruschel Dutra",
    author_email="druscheld@gmail.com",
    url='https://git.cta.if.ufrgs.br/ruschel/ifscube',
    platform='Linux',
    license='GPLv3',
    ext_modules=[ext1],
    classifiers=[
      'Programming Language :: Python :: 3.6'
    ],
)
