from numpy.distutils.core import Extension

ext1 = Extension(name = 'profiles',
                 sources = ['profiles.f90'])

if __name__ == "__main__":

    from numpy.distutils.core import setup

    setup(name = 'lprof',
        description = "Line profile functions",
        author = "Daniel Ruschel Dutra",
        autor_email = "druscheld@gmail.com",
        ext_modules = [ext1]
        )
          
