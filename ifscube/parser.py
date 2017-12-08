import configparser


class LineFitParser:

    def __init__(self, *args, **kwargs):

        self.component_names = []
        self.p0 = []
        self.bounds = []
        self.constraints = []

        if args or kwargs:
            self.__load__(*args, **kwargs)

    def __load__(self, fname, **kwargs):

        self.cfg = configparser.ConfigParser()
        self.cfg.read(fname)

        par_names = dict(
            gaussian=('flux', 'wavelength', 'sigma'),
            gausshermite=('flux', 'wavelength', 'sigma', 'h3', 'h4'),
        )

        self.par_names = par_names[self.cfg['fit']['function']]

        # Each section has to be a line, except for the DEFAULT, MINOPTS,
        # and CONTINUUM sections.
        for line in self.cfg.sections():
            if line not in ['continuum', 'minimization', 'fit']:
                self.component_names += [line]
                self.parse_line(self.cfg[line])

        self._fit()

    def _minimization(self):

        minopts = {**self.cfg['minimization']}
        self.minopts = minopts

    def _fit(self):

        fit_opts = {**self.cfg['fit']}

        boolean_args = ['plotfit', 'fit_continuum', 'writefits']
        for i in boolean_args:
            if i in fit_opts:
                fit_opts[i] = self.cfg.getboolean('fit', i)

        key = 'fitting_window'
        if key in fit_opts:
            a = fit_opts[key].split(':')
            fit_opts['fitting_window'] = tuple([float(i) for i in a])

        self.fit_opts = fit_opts

    def _bounds_parser(self, props):

        if len(props) > 1:
            low, up = [
                float(i) if (i != '') else None for i in props[1].split(':')]
            self.bounds += [(low, up)]
        else:
            self.bounds += [(None, None)]

    def parse_line(self, line):

        for par in self.par_names:
            props = line[par].split(',')
            self.p0 += [float(props[0])]
            self._bounds_parser(props)

    def get_vars(self):

        d = {**vars(self), **self.fit_opts}
        todel = ['cfg', 'component_names', 'par_names', 'fit_opts']
        for i in todel:
            del d[i]
        return d
