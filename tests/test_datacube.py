from importlib import resources
import ifscube


def test_plotspec():
    cube_file = str(resources.files("ifscube") / "../../examples/ngc3081_cube.fits")
    cube = ifscube.Cube(cube_file)
    cube.plotspec(x=(2, 3), y=(4, 5), show=False)


import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from ifscube import Cube

N_LAMBDA, N_Y, N_X = 30, 47, 53  # spatial sizes are deliberately uneven


@pytest.fixture
def cube_file(tmp_path):
    rng = np.random.default_rng(0)
    sci = rng.normal(size=(N_LAMBDA, N_Y, N_X)).astype("float32")
    var = np.abs(sci)
    flags = (sci > 2).astype("int16")

    header = fits.Header()
    for key, value in dict(
            CRPIX1=1, CRPIX2=1, CRPIX3=1,
            CRVAL1=10.0, CRVAL2=-5.0, CRVAL3=5000.0,
            CD1_1=-1e-4, CD2_2=1e-4, CD3_3=2.0,
            CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CTYPE3="WAVE",
            CUNIT3="Angstrom").items():
        header[key] = value

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(sci, header, name="SCI"),
        fits.ImageHDU(var, header, name="VAR"),
        fits.ImageHDU(flags, header, name="FLAGS"),
        fits.ImageHDU(np.arange(N_LAMBDA, dtype="float64"), name="WAVE"),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="a", format="J", array=np.arange(3))], name="VOR"),
    ])
    path = tmp_path / "cube.fits"
    hdul.writeto(path)
    return path


def _reassemble(paths, extension, shape, dtype):
    out = np.zeros(shape, dtype=dtype)
    for path in paths:
        with fits.open(path) as tile:
            y0, x0 = tile[extension].header["TILEY0"], tile[extension].header["TILEX0"]
            data = tile[extension].data
            out[:, y0:y0 + data.shape[1], x0:x0 + data.shape[2]] = data
    return out


@pytest.mark.parametrize("ny, nx", [(1, 1), (3, 4), (47, 1), (1, 53)])
def test_split_reassembles_original(cube_file, tmp_path, ny, nx):
    paths = Cube.split(cube_file, ny=ny, nx=nx, output_dir=tmp_path / "tiles")

    assert len(paths) == ny * nx
    assert all(p.exists() for p in paths)
    for ext in ("SCI", "VAR", "FLAGS"):
        original = fits.getdata(cube_file, ext)
        rebuilt = _reassemble(paths, ext, original.shape, original.dtype)
        np.testing.assert_array_equal(rebuilt, original)


def test_split_tile_sizes_differ_by_at_most_one(cube_file, tmp_path):
    paths = Cube.split(cube_file, ny=3, nx=4, output_dir=tmp_path / "tiles")
    shapes = {fits.getdata(p, "SCI").shape for p in paths}

    assert {s[0] for s in shapes} == {N_LAMBDA}
    assert max(s[1] for s in shapes) - min(s[1] for s in shapes) <= 1
    assert max(s[2] for s in shapes) - min(s[2] for s in shapes) <= 1


def test_split_keeps_wcs_consistent(cube_file, tmp_path):
    full_wcs = WCS(fits.getheader(cube_file, "SCI"))
    for path in Cube.split(cube_file, ny=3, nx=4, output_dir=tmp_path / "tiles"):
        header = fits.getheader(path, "SCI")
        y0, x0 = header["TILEY0"], header["TILEX0"]
        tile_world = WCS(header).all_pix2world([[0, 0, 0]], 0)
        full_world = full_wcs.all_pix2world([[x0, y0, 0]], 0)
        np.testing.assert_allclose(tile_world, full_world)


def test_split_extensions(cube_file, tmp_path):
    path = Cube.split(cube_file, ny=2, nx=2, output_dir=tmp_path / "tiles")[0]
    with fits.open(path) as tile:
        names = [hdu.name for hdu in tile]

    assert names == ["PRIMARY", "SCI", "VAR", "FLAGS", "WAVE"]  # tables are dropped
    assert fits.getdata(path, "WAVE").shape == (N_LAMBDA,)  # 1D extension copied whole


def test_split_output_names_and_default_dir(cube_file):
    paths = Cube.split(cube_file, ny=2, nx=2)

    assert [p.name for p in paths] == [
        "cube_y00_x00.fits", "cube_y00_x01.fits", "cube_y01_x00.fits", "cube_y01_x01.fits"]
    assert all(p.parent == cube_file.parent for p in paths)


@pytest.mark.filterwarnings("ignore:Wavelength read in meters")
def test_split_tile_loads_as_cube(cube_file, tmp_path):
    paths = Cube.split(cube_file, ny=3, nx=4, output_dir=tmp_path / "tiles")
    tile = Cube(str(paths[5]), scidata="SCI", variance="VAR", flags="FLAGS", wcs_axis=3)

    assert tile.data.shape == fits.getdata(paths[5], "SCI").shape
    np.testing.assert_allclose(tile.wl, 5000.0 + 2.0 * np.arange(N_LAMBDA))


def test_split_refuses_existing_files_without_overwrite(cube_file, tmp_path):
    Cube.split(cube_file, ny=2, nx=2, output_dir=tmp_path / "tiles")

    with pytest.raises(OSError):
        Cube.split(cube_file, ny=2, nx=2, output_dir=tmp_path / "tiles")
    Cube.split(cube_file, ny=2, nx=2, output_dir=tmp_path / "tiles", overwrite=True)


@pytest.mark.parametrize("ny, nx", [(0, 1), (1, 0), (N_Y + 1, 1), (1, N_X + 1)])
def test_split_invalid_division(cube_file, tmp_path, ny, nx):
    with pytest.raises(ValueError):
        Cube.split(cube_file, ny=ny, nx=nx, output_dir=tmp_path / "tiles")