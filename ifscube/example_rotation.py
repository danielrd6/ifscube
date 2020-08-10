import matplotlib.pyplot as plt
import numpy as np

from ifscube import models, rotation


def create_fake_data(plot: bool = False):
    y, x = np.indices((100, 100))
    m = models.DiskRotation(amplitude=250.0, c_0=3.0, p=1.25, phi_0=np.deg2rad(60.0), theta=np.deg2rad(30.0), v_sys=0.0,
                            x_0=55, y_0=45)
    data = np.random.normal(m(x, y), 5.0)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(data, origin='lower', cmap='Spectral_r')
        plt.colorbar(im, ax=ax)
        plt.show()

    return data


def main():
    data = create_fake_data(plot=False)

    config = rotation.Config('rotation.ini')

    r = rotation.Rotation(input_data=data)

    config.model['x_0'] = 55
    config.model['y_0'] = 45
    config.model['theta'] = np.deg2rad(30)
    config.model['phi_0'] = np.deg2rad(60)
    r.update_model(config.model)

    r.update_bounds(config.bounds)
    r.update_fixed(config.fixed)
    r.fit_model(maxiter=1000)

    r.print_solution()
    r.plot_results(contours=False)


if __name__ == '__main__':
    main()
