from ifscube import Cube


if __name__ == '__main__':

    cube = Cube('ngc3081_cube.fits')

    mask = [
        [5715, 5730],
        [5870, 5920],
        [6075, 6100],
        [6290, 6325],
        [6355, 6393],
        [6430, 6445],
        [6500, 6640],
        [6675, 6690],
    ]

    cube.ppxf_kinematics(fitting_window=(5600, 6700), plot_fit=True, individual_spec=(3, 3), write_fits=False,
                         mask=mask, deg=3)
