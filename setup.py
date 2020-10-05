from numpy.distutils.core import Extension, setup

with open('./ifscube/.version', 'r') as version_file:
    __version__ = version_file.read().strip('\n')

ext1 = Extension(name='ifscube.elprofile', sources=['ifscube/profiles.f90'])

package_data = {
    'ifscube': [
        'examples/*',
        'data/*',
        'docs/*',
        'tests/*'
    ]
}

setup(
    name='ifscube',
    python_requires='>=3.7',
    version=__version__,
    packages=['ifscube', 'ifscube.io'],
    package_data=package_data,
    scripts=['bin/fit_scrutinizer', 'bin/cubefit', 'bin/specfit', 'bin/fit_rotation'],
    description="Fit emssision lines",
    author="Daniel Ruschel Dutra",
    author_email="daniel@astro.ufsc.br",
    url='https://github.com/danielrd6/ifscube',
    platform='Linux',
    license='GPLv3',
    ext_modules=[ext1],
    classifiers=['Programming Language :: Python :: 3.7'],
)
