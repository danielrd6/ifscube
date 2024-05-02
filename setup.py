from setuptools import Extension, setup
import numpy as np

setup(
    ext_modules=[
        Extension(name="ifscube.elprofile",
                  include_dirs=[np.get_include()],
                  sources=["ifscube/profiles.c"])
    ]
)
