#!/usr/bin/python
import argparse
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ma

from ifscube.io import line_fit


class Scrutinizer:

    def __init__(self, window, args):

        self.fit_status_mask = args.fit_status_mask

        fit = line_fit.load_fit(args.fitfile)

        self.window = window
        self.buttonPlot = tk.Button(window, text='Image plot', command=self.plot)
        self.buttonFollow = tk.Button(window, text='Follow mouse', command=self.follow)
        self.buttonSinglePlot = tk.Button(window, text='Plot on click', command=self.single_plot)

        self.text = tk.Text(window, bg='white', height=10, width=80, font=('Fixedsys', 12))

        l_parameter = tk.Listbox(window, selectmode='single', exportselection=0)
        l_component = tk.Listbox(window, selectmode='single', exportselection=0)

        n_components = len(fit.feature_names)
        n_parameters = fit.parameters_per_feature

        self.parameter_names = fit.parameter_names
        for i in [_[1] for _ in fit.parameter_names[:n_parameters]]:
            l_parameter.insert('end', i)

        for i in ['flux_model', 'flux_direct', 'eqw_model', 'eqw_direct']:
            if hasattr(fit, i):
                l_parameter.insert('end', i)

        self.n_components = n_components
        self.n_parameters = n_parameters

        if hasattr(fit, 'feature_names'):
            for i in fit.feature_names:
                l_component.insert('end', i)
        else:
            for i in range(n_components):
                l_component.insert('end', i)

        if args.small_screen:
            self.fig = Figure(figsize=(3, 3))
            self.fitplot = Figure(figsize=(6, 3))
        else:
            self.fig = Figure(figsize=(6, 6))
            self.fitplot = Figure(figsize=(12, 6))
        self.ax1 = self.fig.add_subplot(111)
        # self.ax2 = self.fitplot.add_subplot(111)

        div = make_axes_locatable(self.ax1)
        self.cax = div.append_axes('right', size='5%', pad=0)
        self.cax.set_xticks([])
        self.cax.set_yticks([])

        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas2 = FigureCanvasTkAgg(self.fitplot, master=self.window)

        canvas.get_tk_widget().grid(row=0, column=0, rowspan=6, columnspan=6)
        canvas2.get_tk_widget().grid(row=0, column=7, rowspan=6, columnspan=12)

        tb_frame = tk.Frame(window)
        tb_frame.grid(row=7, column=0, columnspan=6, sticky='W')
        toolbar = NavigationToolbar2Tk(canvas, tb_frame)
        toolbar.update()

        other_tb_frame = tk.Frame(window)
        other_tb_frame.grid(row=7, column=7, columnspan=12, sticky='W')
        other_toolbar = NavigationToolbar2Tk(canvas2, other_tb_frame)
        other_toolbar.update()

        # Labels
        parameter_label = tk.Label(self.window, text="Fit parameter")
        parameter_label.grid(row=8, column=1)

        component_label = tk.Label(self.window, text="Spectral feature")
        component_label.grid(row=8, column=2)

        n = 9
        self.buttonPlot.grid(row=n, column=0, sticky='W')
        self.buttonFollow.grid(row=n + 1, column=0, sticky='W')
        self.buttonSinglePlot.grid(row=n + 2, column=0, sticky='W')
        l_parameter.grid(row=n, column=1, rowspan=4)
        l_component.grid(row=n, column=2, rowspan=4)
        self.text.grid(row=n, column=7, rowspan=4)
        self.text.insert("insert", "Select a fit parameter and a spectral feature.")

        self.l_par = l_parameter
        self.l_component = l_component

        canvas.draw()

        self.canvas = canvas
        self.canvas2 = canvas2

        self.fit = fit

        self.connect_id = None

    def single_plot(self):
        try:
            self.canvas.mpl_disconnect(self.connect_id)
        except AttributeError:
            pass
        self.connect_id = self.canvas.mpl_connect('button_press_event', self.onclick)

    def follow(self):
        try:
            self.canvas.mpl_disconnect(self.connect_id)
        except AttributeError:
            pass
        self.connect_id = self.canvas.mpl_connect('motion_notify_event', self.onclick)

    def get_image(self):
        try:
            par = self.l_par.curselection()[0]
        except IndexError:
            # noinspection PyUnresolvedReferences
            tk.messagebox.showwarning("Warning!", "You must select a parameter for plotting!")
            return None, None
        try:
            comp = self.l_component.curselection()[0]
        except IndexError:
            # noinspection PyUnresolvedReferences
            tk.messagebox.showwarning("Warning!", "You must select a spectral feature for plotting!")
            return None, None

        parameter = (self.l_component.get(comp), self.l_par.get(par))

        if parameter[1] in ['eqw_model', 'eqw_direct', 'flux_model', 'flux_direct']:
            data = getattr(self.fit, parameter[1])[comp]
        else:
            data = self.fit.solution[self.parameter_names.index(parameter)]

        if parameter[1] in ['velocity', 'h_3', 'h_4']:
            cm = 'Spectral_r'
        else:
            cm = 'inferno'

        if 'eqw' in parameter[1]:
            d = -ma.masked_invalid(data)
        else:
            d = ma.masked_invalid(data)

        if self.fit_status_mask:
            d.mask |= self.fit.fit_status != 0

        return d, cm

    def plot(self, contrast=1):

        a = self.ax1
        a.cla()
        self.cax.cla()

        d, cm = self.get_image()
        if (d is not None) and (cm is not None):
            z0 = np.percentile(d[~d.mask], contrast)
            z1 = np.percentile(d[~d.mask], 100 - contrast)

            if cm == 'Spectral_r':
                im = a.pcolormesh(d, cmap=cm, vmin=z0, vmax=z1)
            else:
                im = a.pcolormesh(d, cmap=cm)

            plt.colorbar(im, cax=self.cax)

            a.set_aspect('equal', 'datalim')

            self.canvas.draw()

    def onclick(self, event):
        if (event.xdata is not None) and (event.ydata is not None):
            i, j = [int(np.floor(x) + 0.5) for x in (event.xdata, event.ydata)]

            self.text.delete('1.0', 'end')
            self.text.insert('insert', '({:6d}, {:6d})\n'.format(i, j))

            s = self.fit.plot(x_0=i, y_0=j, figure=self.fitplot, return_results=True)
            plt.show()
            self.text.insert('insert', s[j, i])

            self.canvas2.draw()
        else:
            self.text.delete('1.0', 'end')
            self.text.insert('insert', 'You clicked outside the plot!')


class OneDFit:

    def __init__(self, file_name):
        spec = line_fit.load_fit(file_name)
        spec.plot()
        plt.show()

        return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('fitfile', type=str, help='Result of the fit')
    parser.add_argument('-m', '--fit-status-mask', help='Apply fit status mask.', action='store_true')
    parser.add_argument('-s', '--single-spectrum', action='store_true',
                        help='Use this option for one-dimensional spectra.')
    parser.add_argument('--small-screen', action='store_true', help='Makes a small window for small screens.')
    arguments = parser.parse_args()

    if arguments.single_spectrum:
        OneDFit(arguments.fitfile)
    else:
        app_window = tk.Tk()
        app_window.title('IFSCUBE Fit Scrutinizer')
        start = Scrutinizer(app_window, arguments)
        app_window.mainloop()
