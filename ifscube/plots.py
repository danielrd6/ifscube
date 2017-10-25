import numpy as np
import ifscube.spectools as st
import matplotlib.pyplot as plt
from astropy.io import fits

def w80(x):

    # Create figure and axes.
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111)

    for i, j in enumerate(x):

        sc = 'C{:d}'.format(i)
        # Plot the spectrum.
        ax.plot(j[2], j[3], c=sc)
    
        # Draw w80 lines.
        for k in j[:2]:
            ax.axvline(k, ls='dashed', color=sc)
    
        # Draw baseline
        ax.axhline(0, ls='dashed', color=sc, alpha=.5)
    
        # Labels.
        ax.set_xlabel(r'Velocity (km/s)')
        ax.set_ylabel(r'Normalized flux units')
    
        ax.minorticks_on()
    
    # Show plot.
    plt.show()
