from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants, units
from astropy.units import Quantity
from astropy.units.equivalencies import doppler_relativistic
from matplotlib import patheffects
from numpy import ma
from scipy.integrate import trapz

from . import Cube
from . import cubetools


class ChannelMaps:
    def __init__(self, cube: Cube, fitting_window: Iterable, method: str = "trapezoidal"):
        """
        Channel maps.

        Parameters
        ----------
        cube : ifscube.Cube
            Data cube with observations.
        method : str
            'trapezoidal' - trapezoidal integration, with boundaries dependent on wavelength sampling.
            'linear_interpolation' - quadrature integration of a linearly interpolated version of the observed spectra.
        """
        self.center_velocities = None
        self.cube = cube
        self.continuum = None
        self.velocity_boundaries = None
        self.wavelength_boundaries = None
        self.wavelength_units = cube.wcs.world_axis_units[0]

        fw = fitting_window
        fw = [_.to(self.wavelength_units).value for _ in fw]
        self.wavelength_mask = (cube.rest_wavelength >= fw[0]) & (cube.rest_wavelength <= fw[1])
        self.wavelength = Quantity(value=cube.rest_wavelength[self.wavelength_mask], unit=self.wavelength_units)
        self.fitting_window = fw

        self.data = self.cube.data[self.wavelength_mask]

        assert method in ["trapezoidal", "linear_interpolation"], f"Unsupported method: {method}."
        self.method = method

    def set_channel_boundaries(self, reference_wavelength, vel_min, vel_max, channels):
        velocities = np.linspace(vel_min, vel_max, channels)
        wavelengths = Quantity(velocities).to(
            unit=reference_wavelength.unit, equivalencies=doppler_relativistic(reference_wavelength))

        if self.method == "trapezoidal":
            closest_wavelengths = []
            for w in wavelengths:
                cw = Quantity(value=self.cube.rest_wavelength, unit=self.cube.wcs.world_axis_units[0]) - w
                closest_wavelengths.append(self.cube.rest_wavelength[np.abs(cw).argsort()][0])

            wb = Quantity(np.array(closest_wavelengths), unit=reference_wavelength.unit)
            vb = wb.to(unit="km / s", equivalencies=doppler_relativistic(reference_wavelength))
            cv = Quantity(value=[(vb[_] + vb[_ + 1]) / 2.0 for _ in range(len(vb) - 1)])
        elif self.method == "linear_interpolation":
            wb = wavelengths
            vb = velocities
            cv = Quantity(value=[(vb[_] + vb[_ + 1]) / 2.0 for _ in range(len(vb) - 1)])
        else:
            raise RuntimeError

        self.wavelength_boundaries = wb
        self.velocity_boundaries = vb
        self.center_velocities = cv

    def set_continuum(self, options: dict = None):
        if options is None:
            options = {'n_iterate': 3, 'degree': 1, 'upper_threshold': 1, 'lower_threshold': 3,
                       'output': 'function'}

        cp = options

        fw = self.fitting_window
        self.continuum = self.cube.continuum(fitting_window=fw, continuum_options=cp)

    def evaluate_channel_maps(self, lambda0, vel_min, vel_max, channels, continuum_options=None):

        self.set_channel_boundaries(reference_wavelength=lambda0, vel_min=vel_min, vel_max=vel_max, channels=channels)
        self.set_continuum(options=continuum_options)
        self.integrate_maps()

        pass

    def integrate_maps(self):
        if self.method == "trapezoidal":
            wl = self.wavelength
            wb = self.wavelength_boundaries
            masks = [(wl >= wb[_]) & (wl <= wb[_ + 1]) for _ in range(len(wb) - 1)]
            cm = [trapz(self.data[_], wl[_], axis=0) for _ in masks]
            print(cm)
        if self.method == "linear_interpolation":
            pass

    def plot_spectrum(self, x: int, y: int):

        data = self.cube.data[self.wavelength_mask, y, x]
        continuum = self.cube.cont[:, y, x]
        wl = self.cube.rest_wavelength[self.wavelength_mask]
        fig, ax = plt.subplots()
        ax.plot(wl, continuum)
        ax.plot(wl, data)
        plt.show()


def channelmaps(cube, lambda0, vel_min, vel_max, channels=6, continuum_width=300, continuum_options=None,
                log_flux=False, angular_scale=None, scale_bar=None, north_arrow=None, lower_threshold=1e-16,
                plot_opts={}, fig_opts={}, width_space=None, height_space=None, text_color='black',
                stroke_color='white', color_bar=True, center_mark=True, screen=True, xy_coords=None):
    """
    Creates velocity channel maps from a data cube.

    Parameters
    ----------
    lambda0: number
        Central wavelength of the desired spectral feature.
    vel_min: number
        Mininum velocity in kilometers per second.
    vel_max: number
        Maximum velocity in kilometers per second.
    channels: int
        Number of channel maps to build.
    continuum_width: number
        Width in wavelength units for the continuum evaluation
        window.
    continuum_options: dict
        Dicitionary of options to be passed to the
        spectools.continuum function.
    log_flux: bool
        If True, takes the base 10 logarithm of the fluxes.
    lower_threshold: number
        Minimum emission flux for plotting, after subtraction
        of the continuum level. Spaxels with flux values below
        lowerThreshold will be masked in the channel maps.
    angular_scale: number
        The angular pixel scale, in arcsec/pix. By default it is readen
        from the header keyword CD1_1.
    scale_bar: dict
        Places a scale bar with the size 'scale_size' in the y,x
        position 'scale_pos', in the first panel, labeled with text
        'scale_tex'.
    north_arrow: dict
        Places reference arrows where north PA is 'north_pa' and east
        is rotated 90 degrees counterclockwise (when 'east_side' is 1)
        or clockwise (when 'east_side' is -1). The arrows have origin
        at position 'arrow_pos'.
    color_bar: bool
        If True draws a colorbar.
    center_mark: bool
        If True, evaluates the continuum centroid and marks it with
        'plus' sign.
    plot_opts: dict
        Dictionary of options to be passed to **pcolormesh**.
    fig_opts: dict
        Options passed to **pyplot.figure**.
    width_space: float
        Horizontal gap between channel maps.
    height_space: float
        Vertical gap between channel maps.
    text_color: matplotlib.color
        The color of the annotated texts specifying the velocity
        bin, the scale bar and scale text and the reference arrows and
        text.
    stroke_color: matplotlib.color
        The color of the the thin stroke drawn around texts and lines to
        increase contrast when those symbols appear over image areas of
        similar color.
    screen: bool
        If screen is True the channel maps are shown on screen.
    xy_coords : tuple
        Tuple with (x, y) coordinates for the plots. If *None*
        coordinates will be automatically calculated based on the
        central pixel of the image.

    Returns
    -------
    """

    sigma = lower_threshold

    # Converting from velocities to wavelength
    wlmin, wlmax = lambda0 * (np.array([vel_min, vel_max]) / constants.c.to(units.km / units.s).value + 1.)

    wlstep = (wlmax - wlmin) / channels
    wl_limits = np.arange(wlmin, wlmax + wlstep, wlstep)

    side = int(np.ceil(np.sqrt(channels)))  # columns
    otherside = int(np.ceil(channels / side))  # lines
    fig = plt.figure(**fig_opts)
    plt.clf()

    if continuum_options is None:
        continuum_options = {'n_iterate': 3, 'degree': 5, 'upper_threshold': 3, 'lower_threshold': 3,
                             'output': 'function'}

    cp = continuum_options
    cw = continuum_width
    fw = lambda0 + np.array([-cw / 2., cw / 2.])

    cube.cont = cube.continuum(
        write_fits=False, output_image=None, fitting_window=fw, continuum_options=cp)

    contwl = cube.rest_wavelength[(cube.rest_wavelength > fw[0]) & (cube.rest_wavelength < fw[1])]
    maps = []
    axes = []
    pmaps = []
    x_axes_0 = []
    x_axes_1 = []
    y_axes_0 = []
    y_axes_1 = []
    mpl.rcParams['xtick.labelsize'] = (12 - int(np.sqrt(channels)))
    mpl.rcParams['ytick.labelsize'] = (12 - int(np.sqrt(channels)))
    mpl.rcParams['figure.subplot.left'] = 0.05
    mpl.rcParams['figure.subplot.right'] = 0.80
    if angular_scale is None:
        try:
            pScale = abs(cube.header['CD1_1'])
        except KeyError:
            print(
                'WARNING! Angular scale \'CD1_1\' not found in the image' +
                'header. Adopting angular scale = 1.')
            pScale = 1.
    else:
        pScale = angular_scale

    for i in np.arange(channels):
        ax = fig.add_subplot(otherside, side, i + 1)
        axes += [ax]
        wl = cube.rest_wavelength
        wl0, wl1 = wl_limits[i], wl_limits[i + 1]
        print(wl[(wl > wl0) & (wl < wl1)])
        wlc, wlwidth = np.average([wl0, wl1]), (wl1 - wl0)

        f_obs = cube.wlprojection(
            wl0=wlc, fwhm=wlwidth, filtertype='box')
        f_cont = cubetools.wlprojection(
            arr=cube.cont, wl=contwl, wl0=wlc, fwhm=wlwidth,
            filtertype='box')
        f = f_obs - f_cont

        mask = (f < sigma) | np.isnan(f)
        channel = ma.array(f, mask=mask)

        if log_flux:
            channel = np.log10(channel)
        if i == 0:
            coords = cube.peak_coords(
                wl_center=lambda0, wl_width=cw, center_type='peak_cen')

        if xy_coords is None:
            y, x = pScale * (np.indices(np.array(f.shape) + 1) - 0.5)
            y, x = y - coords[0] * pScale, x - coords[1] * pScale
        else:
            x, y = xy_coords
        if center_mark:
            mpl.pyplot.plot(0, 0, 'w+', lw=3)
            mpl.pyplot.plot(0, 0, 'k+', lw=2)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        pmap = ax.pcolormesh(x, y, channel, **plot_opts)
        ax.set_aspect('equal', 'datalim')

        if scale_bar is not None:
            scale_text = scale_bar['scale_text']
            scale_size = scale_bar['scale_size']
            scale_pos = scale_bar['scale_pos']
            scale_panel = scale_bar['scale_panel']
            if i == scale_panel:
                pos_y, pos_x = (
                    (y.max() - y.min()) * scale_pos[0] + y.min(),
                    (x.max() - x.min()) * scale_pos[1] + x.min())
                delt_y = (y.max() - y.min()) * 0.05
                ax.plot(
                    [pos_x, pos_x + scale_size], [pos_y, pos_y], '-',
                    lw=3, color=text_color,
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=4.5, foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])
                ax.annotate(
                    scale_text, xy=(
                        pos_x + scale_size / 2., pos_y + delt_y),
                    ha='center',
                    va='bottom',
                    color=text_color,
                    weight='bold',
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=1.5,
                            foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])

        if north_arrow is not None:
            north_pa = 90. + north_arrow['north_pa']
            east_side = north_arrow['east_side']
            arrow_pos = np.array(north_arrow['arrow_pos']) * \
                        np.array([(y.max() - y.min()), (x.max() - x.min())]) + \
                        np.array([y.min(), x.min()])
            n_panel = north_arrow['n_panel']
            if (i == n_panel):
                arrSize = 0.2 * \
                          np.sqrt(
                              (y.max() - y.min()) ** 2 + (x.max() - x.min()) ** 2)
                y_north = arrow_pos[0] + arrSize * np.sin(
                    np.deg2rad(north_pa))
                x_north = arrow_pos[1] + arrSize * np.cos(
                    np.deg2rad(north_pa))
                y_east = arrow_pos[0] + arrSize * np.sin(
                    np.deg2rad(north_pa + east_side * 90.))
                x_east = arrow_pos[1] + arrSize * np.cos(
                    np.deg2rad(north_pa + east_side * 90.))
                plt.plot(
                    [x_north, arrow_pos[1], x_east],
                    [y_north, arrow_pos[0], y_east], '-',
                    lw=3, color=text_color,
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=4.5,
                            foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])
                l_ang = -15
                y_letter_n = arrow_pos[0] + 1.15 * arrSize * np.sin(
                    np.deg2rad(north_pa + l_ang))
                x_letter_n = arrow_pos[1] + 1.15 * arrSize * np.cos(
                    np.deg2rad(north_pa + l_ang))
                y_letter_e = arrow_pos[0] + 1.15 * arrSize * np.sin(
                    np.deg2rad(north_pa + 90. + l_ang))
                x_letter_e = arrow_pos[1] + 1.15 * arrSize * np.cos(
                    np.deg2rad(north_pa + 90. + l_ang))
                plt.text(
                    x_letter_n, y_letter_n, 'N',
                    ha='center',
                    va='center',
                    weight='bold',
                    color=text_color,
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=1.5,
                            foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])
                plt.text(
                    x_letter_e, y_letter_e, 'E',
                    ha='center',
                    va='center',
                    weight='bold',
                    color=text_color,
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=1.5,
                            foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])

        ax.annotate(
            '{:.0f}'.format((wlc - lambda0) / lambda0 * 2.99792e+5),
            xy=(0.1, 0.8), xycoords='axes fraction', weight='bold',
            color=text_color, path_effects=[
                patheffects.Stroke(
                    linewidth=1.5, foreground=stroke_color, alpha=0.3),
                patheffects.Normal()])
        if i % side != 0:
            ax.set_yticklabels([])
        if i / float((otherside - 1) * side) < 1:
            ax.set_xticklabels([])
        maps += [channel]
        pmaps += [pmap]
        x_axes_0 = np.append(x_axes_0, ax.get_position().x0)
        x_axes_1 = np.append(x_axes_1, ax.get_position().x1)
        y_axes_0 = np.append(y_axes_0, ax.get_position().y0)
        y_axes_1 = np.append(y_axes_1, ax.get_position().y1)

    fig.subplots_adjust(wspace=width_space, hspace=height_space)

    if color_bar:
        # x_min_axes_0 = np.min(x_axes_0)
        x_max_axes_1 = np.max(x_axes_1)
        y_min_axes_0 = np.min(y_axes_0)
        y_max_axes_1 = np.max(y_axes_1)

        # setup colorbar axes.
        ax1 = fig.add_axes([
            x_max_axes_1, y_min_axes_0, 0.015,
            y_max_axes_1 - y_min_axes_0])
        fig.colorbar(pmap, cax=ax1, orientation='vertical')

    if screen is True:
        plt.show()

    return maps, axes, pmaps


def rgb_line_compose(
        cube, lambdas, velmin, velmax, channels=9, continuum_width=300,
        continuum_opts=None, logFlux=True, angScale=None, scaleBar={},
        northArrow={}, lowerThreshold=1e-16, plot_opts={}, fig_opts={},
        wspace=None, hspace=None, text_color='black', stroke_color='white',
        center_mark=True, bg_color=(0.25, 0.25, 0.25)):
    """
    Creates velocity channel maps from a data cube for two or three
    different lines and compose them as an RGB image attributing each
    channel map to one of the RGB channels.

    Parameters
    ----------
    lambdas : list
        Central wavelength of the desired spectral feature.
    vmin : number
        Mininum velocity in kilometers per second.
    vmax : number
        Maximum velocity in kilometers per second.
    channels : integer
        Number of channel maps to build.
    continuum_width : number
        Width in wavelength units for the continuum evaluation
        window.
    continuum_opts : dictionary
        Dicitionary of options to be passed to the
        spectools.continuum function.
    logFlux : Boolean
        If True, takes the base 10 logarithm of the fluxes.
    lowerThreshold: number
        Minimum emission flux for plotting, after subtraction
        of the continuum level. Spaxels with flux values below
        lowerThreshold will be masked in the channel maps.
    angScale : number
        The angular pixel scale, in arcsec/pix. By default it is readen
        from the header keyword CD1_1.
    scaleBar : dictionary
        Places a scale bar with the size 'scale_size' in the y,x
        position 'scale_pos', in the first panel, labeled with text
        'scale_tex'.
    northArrow : dictionary
        Places reference arrows where north PA is 'north_pa' and east
        is rotated 90 degrees counterclockwise (when 'east_side' is 1)
        or clockwise (when 'east_side' is -1). The arrows have origin
        at position 'arrow_pos'.
    center_mark : Boolean
        If True, evaluates the continuum centroid and marks it with
        'plus' sign.
    plot_opts : dict
        Dictionary of options to be passed to **pcolormesh**.
    fig_opts : dict
        Options passed to **pyplot.figure**.
    wspace : number
        Horizontal gap between channel maps.
    hspace : number
        Vertical gap between channel maps.
    text_color : matplotlib color
        The color of the annotated text specifying the velocity
        bin.
    stroke_color : matplotlibcolor
        The color of the the thin stroke drawn around texts and lines to
        increase contrast when those symbols appear over image areas of
        similar color.
    bg_color : matplotlibcolor
        The color of the background area, where data is missing.
    """

    sigma = lowerThreshold

    # Converting from velocities to wavelength
    wlmin_r, wlmax_r = lambdas[0] * (
            np.array([velmin, velmax]) /
            constants.c.to(units.km / units.s).value + 1.
    )
    wlmin_g, wlmax_g = lambdas[1] * (
            np.array([velmin, velmax]) /
            constants.c.to(units.km / units.s).value + 1.
    )

    wl_limits_r = np.linspace(wlmin_r, wlmax_r, channels + 1)
    wl_limits_g = np.linspace(wlmin_g, wlmax_g, channels + 1)

    side = int(np.ceil(np.sqrt(channels)))  # columns
    otherside = int(np.ceil(channels / side))  # lines
    fig = plt.figure(**fig_opts)
    plt.clf()

    if continuum_opts is None:
        continuum_opts = dict(
            niterate=3, degr=5, upper_threshold=3, lower_threshold=3,
            returns='function')

    cp = continuum_opts
    cw = continuum_width
    fw_r = lambdas[0] + np.array([-cw / 2., cw / 2.])
    fw_g = lambdas[1] + np.array([-cw / 2., cw / 2.])

    cube.cont_r = cube.continuum(
        write_fits=False, output_image=None, fitting_window=fw_r, continuum_options=cp)
    cube.cont_g = cube.continuum(
        write_fits=False, output_image=None, fitting_window=fw_g, continuum_options=cp)

    contwl_r = cube.rest_wavelength[(cube.rest_wavelength > fw_r[0]) & (cube.rest_wavelength < fw_r[1])]
    contwl_g = cube.rest_wavelength[(cube.rest_wavelength > fw_g[0]) & (cube.rest_wavelength < fw_g[1])]
    channelMaps = []
    axes = []
    pmaps = []
    x_axes_0 = []
    x_axes_1 = []
    y_axes_0 = []
    y_axes_1 = []
    mpl.rcParams['xtick.labelsize'] = (12 - int(np.sqrt(channels)))
    mpl.rcParams['ytick.labelsize'] = (12 - int(np.sqrt(channels)))
    mpl.rcParams['figure.subplot.left'] = 0.05
    mpl.rcParams['figure.subplot.right'] = 0.80
    if angScale is None:
        try:
            pScale = abs(cube.header['CD1_1'])
        except KeyError:
            print(
                'WARNING! Angular scale \'CD1_1\' not found in the image' +
                'header. Adopting angular scale = 1. arcsec/pix')
            pScale = 1.
    else:
        pScale = angScale

    if (len(lambdas) == 3):
        wlmin_b, wlmax_b = lambdas[2] * (
                np.array([velmin, velmax]) /
                constants.c.to(units.km / units.s).value + 1.
        )
        wl_limits_b = np.linspace(wlmin_b, wlmax_b, channels + 1)
        fw_b = lambdas[2] + np.array([-cw / 2., cw / 2.])
        cube.cont_b = cube.continuum(
            write_fits=False, output_image=None, fitting_window=fw_b, continuum_options=cp)
        contwl_b = cube.rest_wavelength[(cube.rest_wavelength > fw_b[0]) & (cube.rest_wavelength < fw_b[1])]

    for i in np.arange(channels):
        ax = fig.add_subplot(otherside, side, i + 1)
        axes += [ax]
        wl = cube.rest_wavelength
        wl0_r, wl1_r = wl_limits_r[i], wl_limits_r[i + 1]
        wl0_g, wl1_g = wl_limits_g[i], wl_limits_g[i + 1]
        print(wl[(wl > wl0_r) & (wl < wl1_r)])
        print(wl[(wl > wl0_g) & (wl < wl1_g)])
        wlc_r, wlwidth_r = np.average([wl0_r, wl1_r]), (wl1_r - wl0_r)
        wlc_g, wlwidth_g = np.average([wl0_g, wl1_g]), (wl1_g - wl0_g)

        f_obs_r = cube.wlprojection(
            wl0=wlc_r, fwhm=wlwidth_r, filtertype='box')
        f_cont_r = cubetools.wlprojection(
            arr=cube.cont_r, wl=contwl_r, wl0=wlc_r, fwhm=wlwidth_r,
            filtertype='box')
        f_r = f_obs_r - f_cont_r

        f_obs_g = cube.wlprojection(
            wl0=wlc_g, fwhm=wlwidth_g, filtertype='box')
        f_cont_g = cubetools.wlprojection(
            arr=cube.cont_g, wl=contwl_g, wl0=wlc_g, fwhm=wlwidth_g,
            filtertype='box')
        f_g = f_obs_g - f_cont_g

        mask_r = (f_r < sigma) | np.isnan(f_r)
        channel_r = ma.array(f_r, mask=mask_r)
        mask_g = (f_g < sigma) | np.isnan(f_g)
        channel_g = ma.array(f_g, mask=mask_g)

        mask_alpha = (mask_r | mask_g)
        channel_alpha = ma.ones(np.shape(channel_r))
        channel_alpha.mask = mask_alpha
        channel_alpha = ma.filled(channel_alpha, fill_value=0)

        if logFlux:
            channel_r = np.log10(channel_r)
            channel_g = np.log10(channel_g)

        if i == 0:
            coords = cube.peak_coords(
                wl_center=lambdas[0], wl_width=cw, center_type='peak_cen')
        y, x = pScale * (np.indices(np.array(f_r.shape) + 1) - 0.5)
        y, x = y - coords[0] * pScale, x - coords[1] * pScale

        if (len(lambdas) == 3):
            wl0_b, wl1_b = wl_limits_b[i], wl_limits_b[i + 1]
            print(wl[(wl > wl0_b) & (wl < wl1_b)])
            wlc_b, wlwidth_b = np.average([wl0_b, wl1_b]), (wl1_b - wl0_b)

            f_obs_b = cube.wlprojection(
                wl0=wlc_b, fwhm=wlwidth_b, filtertype='box')
            f_cont_b = cubetools.wlprojection(
                arr=cube.cont_b, wl=contwl_b, wl0=wlc_b, fwhm=wlwidth_b,
                filtertype='box')
            f_b = f_obs_b - f_cont_b

            mask_b = (f_b < sigma) | np.isnan(f_b)
            channel_b = ma.array(f_b, mask=mask_b)
            mask_alpha = (mask_alpha | mask_b)
            channel_alpha = ma.ones(np.shape(channel_r))
            channel_alpha.mask = mask_alpha
            channel_alpha = ma.filled(channel_alpha, fill_value=0)

            if logFlux:
                channel_b = np.log10(channel_b)
            chan_min_b = ma.min(channel_b)
            chan_max_b = ma.max(channel_b)
            channel_b = (channel_b - chan_min_b) / (
                    chan_max_b - chan_min_b)
        elif (len(lambdas) == 2):
            channel_b = np.zeros(np.shape(channel_r))

        chan_min_r = ma.min(channel_r)
        chan_max_r = ma.max(channel_r)
        chan_min_g = ma.min(channel_g)
        chan_max_g = ma.max(channel_g)
        channel_r = (channel_r - chan_min_r) / (chan_max_r - chan_min_r)
        channel_g = (channel_g - chan_min_g) / (chan_max_g - chan_min_g)

        channel = np.ma.dstack(
            (channel_r, channel_g, channel_b, channel_alpha))
        ax.set_facecolor(bg_color)
        imap = ax.imshow(
            channel, extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower')
        ax.set_aspect('equal', 'datalim')
        if center_mark:
            mpl.pyplot.plot(0, 0, 'w+', lw=3)
            mpl.pyplot.plot(0, 0, 'k+', lw=2)

        if (scaleBar != {}):
            scale_text = scaleBar['scale_text']
            scale_size = scaleBar['scale_size']
            scale_pos = scaleBar['scale_pos']
            scale_panel = scaleBar['scale_panel']
            if (i == scale_panel):
                pos_y, pos_x = (
                    (y.max() - y.min()) * scale_pos[0] + y.min(),
                    (x.max() - x.min()) * scale_pos[1] + x.min())
                delt_y = (y.max() - y.min()) * 0.05
                ax.plot(
                    [pos_x, pos_x + scale_size], [pos_y, pos_y], '-',
                    lw=3,
                    color=text_color,
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=4.5,
                            foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])
                ax.annotate(
                    scale_text, xy=(
                        pos_x + scale_size / 2., pos_y + delt_y),
                    ha='center',
                    va='bottom',
                    color=text_color,
                    weight='bold',
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=1.5,
                            foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])

        if (northArrow != {}):
            north_pa = 90. + northArrow['north_pa']
            east_side = northArrow['east_side']
            arrow_pos = np.array(northArrow['arrow_pos']) * \
                        np.array([(y.max() - y.min()), (x.max() - x.min())]) + \
                        np.array([y.min(), x.min()])
            n_panel = northArrow['n_panel']
            if (i == n_panel):
                arrSize = 0.2 * \
                          np.sqrt(
                              (y.max() - y.min()) ** 2 + (x.max() - x.min()) ** 2)
                y_north = arrow_pos[0] + arrSize * np.sin(
                    np.deg2rad(north_pa))
                x_north = arrow_pos[1] + arrSize * np.cos(
                    np.deg2rad(north_pa))
                y_east = arrow_pos[0] + arrSize * np.sin(
                    np.deg2rad(north_pa + east_side * 90.))
                x_east = arrow_pos[1] + arrSize * np.cos(
                    np.deg2rad(north_pa + east_side * 90.))
                plt.plot(
                    [x_north, arrow_pos[1], x_east],
                    [y_north, arrow_pos[0], y_east], '-',
                    lw=3, color=text_color,
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=4.5, foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])
                l_ang = -15
                y_letter_n = arrow_pos[0] + 1.15 * arrSize * np.sin(
                    np.deg2rad(north_pa + l_ang))
                x_letter_n = arrow_pos[1] + 1.15 * arrSize * np.cos(
                    np.deg2rad(north_pa + l_ang))
                y_letter_e = arrow_pos[0] + 1.15 * arrSize * np.sin(
                    np.deg2rad(north_pa + 90. + l_ang))
                x_letter_e = arrow_pos[1] + 1.15 * arrSize * np.cos(
                    np.deg2rad(north_pa + 90. + l_ang))
                plt.text(
                    x_letter_n, y_letter_n, 'N',
                    ha='center',
                    va='center',
                    weight='bold',
                    color=text_color,
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=1.5, foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])
                plt.text(
                    x_letter_e, y_letter_e, 'E',
                    ha='center',
                    va='center',
                    weight='bold',
                    color=text_color,
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=1.5, foreground=stroke_color,
                            alpha=0.3),
                        patheffects.Normal()])

        ax.annotate(
            '{:.0f}'.format(
                (wlc_r - lambdas[0]) / lambdas[0] * 2.99792e+5),
            xy=(0.1, 0.8), xycoords='axes fraction', weight='bold',
            color=text_color, path_effects=[
                patheffects.Stroke(
                    linewidth=1.5, foreground=stroke_color, alpha=0.3),
                patheffects.Normal()])
        if i % side != 0:
            ax.set_yticklabels([])
        if i / float((otherside - 1) * side) < 1:
            ax.set_xticklabels([])
        channelMaps += [channel]
        pmaps += [imap]
        x_axes_0 = np.append(x_axes_0, ax.get_position().x0)
        x_axes_1 = np.append(x_axes_1, ax.get_position().x1)
        y_axes_0 = np.append(y_axes_0, ax.get_position().y0)
        y_axes_1 = np.append(y_axes_1, ax.get_position().y1)

    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.show()

    return channelMaps, axes, pmaps
