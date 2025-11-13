import matplotlib.pyplot as plt
import numpy as np


def velocity_width(results: dict):
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111)

    power = np.int(np.log10(np.percentile(results['direct_spectrum'], 90)))

    for i, j in enumerate(['model', 'direct']):

        sc = 'C{:d}'.format(i)

        ax.plot(results[f'{j}_velocities'], results[f'{j}_spectrum'], c=sc)

        for k in ['lower', 'upper']:
            ax.axvline(results[f'{j}_{k}_velocity'], ls=['dashed', 'dotted'][i], color=sc)

        ax.axhline(0, ls='dotted', color=sc, alpha=.5)

        ax.set_xlabel('Velocity (km/s)')
        if power != 0:
            ax.set_ylabel(r'$F_\lambda \times 10^{{{:d}}}$'.format(power))
        else:
            ax.set_ylabel(r'$F_\lambda$')

        ax.minorticks_on()

    title = ', '.join(
        [f'$W_{{\\rm {i}}} = {results[f"{j}_velocity_width"]:.0f}\\,{{\\rm km\\,s^{{-1}}}}$'
         for i, j in zip('MD', ['model', 'direct'])])

    ax.set_title(title)
    plt.show()
