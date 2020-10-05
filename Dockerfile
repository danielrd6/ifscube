FROM python:3.7

WORKDIR /ifscube
ADD . /ifscube/
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends gfortran
RUN pip install numpy
RUN pip install .
WORKDIR /ifscube/ifscube/examples/
RUN specfit --overwrite -c halpha.cfg manga_onedspec.fits
RUN cubefit --overwrite -c halpha_cube.cfg ngc3081_cube.fits
RUN cubefit --overwrite -c halpha_cube.cfg ngc3081_cube.bin.fits
