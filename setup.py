from numpy.distutils.core import Extension, setup


with open('./ifscube/.version', 'r') as verfile:
    __version__ = verfile.read().strip('\n')

ext1 = Extension(name='ifscube.elprofile',
                 sources=['ifscube/profiles.f90'])

setup(name='ifscube',
      version=__version__,
      packages=['ifscube'],
      scripts=['bin/fit_scrutinizer'],
      description="Fit emssision lines",
      author="Daniel Ruschel Dutra",
      author_email="druscheld@gmail.com",
      url='https://git.cta.if.ufrgs.br/ruschel/ifscube',
      platform='Linux',
      license='GPLv3',
      ext_modules=[ext1])
