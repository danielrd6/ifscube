from setuptools import setup
from setuptools import Extension

ext1 = Extension(name='ifscube.elprofile', sources=['ifscube/profiles.c'])

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
    python_requires='>=3.12',
    version="1.2",
    packages=['ifscube', 'ifscube.io'],
    package_data=package_data,
    scripts=['bin/fit_scrutinizer.py', 'bin/cubefit', 'bin/specfit', 'bin/fit_rotation'],
    description="Fits emssision lines",
    author="Daniel Ruschel Dutra",
    author_email="daniel.ruschel@.ufsc.br",
    url='https://github.com/danielrd6/ifscube',
    platform='Linux',
    license='GPLv3',
    ext_modules=[ext1],
    classifiers=['Programming Language :: Python :: 3.12'],
)
