import configparser
import copy


class ConstraintParser:

    def __init__(self, expr, linefit=None):

        self.linefit = linefit
        self.expr = expr
        self._operators = '+-*/><'
        self._containers = '()[]{}'
        self._spacer = ' \t\n\r'

        self.__limits__()
        self.__classify__()
        self._getConstraintType()

    def __what__(self, c):

        if c.isalpha():
            group = 'alpha'
        elif c.isdecimal():
            group = 'number'
        elif c in self._operators:
            group = 'operator'
        elif c in self._spacer:
            group = 'spacer'
        else:
            group = 'unkonwn'

        return group

    def __limits__(self):

        previous = ''
        lim = []

        for i, c in enumerate(self.expr):

            current = self.__what__(c)
            if current != previous:
                lim += [i]
                previous = copy.copy(current)
            else:
                continue

        self.lim = lim

    def __classify__(self):

        operators = []
        variables = []
        containers = []
        numbers = []

        for c in self.tolist():

            if self.__what__(c) == 'alpha':
                variables += [c]
            elif self.__what__(c) == 'number':
                numbers += [c]
            elif self.__what__(c) == 'operator':
                operators += [c]
            elif self.__what__(c) == 'container':
                containers += [c]
            elif self.__what__(c) == 'unknown':
                continue

        self.numbers = numbers
        self.operators = operators
        self.variables = variables
        self.containers = containers

    def __idx__(self, cname, pname):

        ci = self.linefit.component_names.index(cname)
        pi = self.linefit.par_names.index(pname)
        npars = len(self.linefit.par_names)

        idx = ci * npars + pi

        return idx

    def _getConstraintType(self):

        lis = self.tolist()

        if lis[0] in '<>':
            t = 'ineq'
        else:
            t = 'eq'

        self.type = t

    def evaluate(self, component, parameter):

        idx = self.__idx__(component, parameter)

        lis = self.tolist()

        if lis[0] in '<>':
            lis.remove(lis[0])

        if (lis[0] in self.numbers) and len(lis) == 1:

            a = float(lis[0])
            def func(x):
                r = a - x[idx]
                return r

        # elif (lis[0] in variables) and len(lis) == 1:

        #     a =

        self.constraint = dict(type=self.type, fun=func)

    def tolist(self):

        elements = []
        lims = self.lim

        for i, j in enumerate(lims):
           
            if j == lims[-1]:
                c = self.expr[lims[i]:None] 
            else:
                c = self.expr[lims[i]:lims[i + 1]]

            if c not in self._spacer:
                elements += [c]

        return elements


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
        self.component_names = [
            i for i in self.cfg.sections() if i not in [
                'continuum', 'minimization', 'fit']
        ]

        # Each section has to be a line, except for the DEFAULT, MINOPTS,
        # and CONTINUUM sections.
        for line in self.component_names:
            self.parse_line(self.cfg[line])

        for line in self.component_names:
            for par in self.par_names:
                prop = self.cfg[line][par].split(',')
                self._constraints(prop, line, par)

        if 'fit' in self.cfg.sections():
            self._fit()
        else:
            self.fit_opts = {}
        if 'minimization' in self.cfg.sections():
            self._minimization()
        else:
            self.minopts = {}
        if 'continuum' in self.cfg.sections():
            self._continuum()
        else:
            self.copts = {}

    def _parse_dict(self, section, float_args, int_args, bool_args):

        d = {**self.cfg[section]}

        for i in float_args:
            if i in d:
                d[i] = self.cfg.getfloat(section, i)

        for i in int_args:
            if i in d:
                d[i] = self.cfg.getint(section, i)

        for i in bool_args:
            if i in d:
                d[i] = self.cfg.getboolean(section, i)

        return d

    def _minimization(self):

        self.minopts = self._parse_dict(
            section='minimization',
            float_args=['eps', 'ftol'],
            int_args=['maxiter'],
            bool_args=['disp'],
        )

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

    def _continuum(self):

        self.copts = self._parse_dict(
            section='continuum',
            float_args=['lower_threshold', 'upper_threshold'],
            int_args=['degr', 'niterate'],
            bool_args=[],
        )

    def _bounds(self, props):

        if len(props) > 1:
            low, up = [
                float(i) if (i != '') else None for i in props[1].split(':')]
            self.bounds += [(low, up)]
        else:
            self.bounds += [(None, None)]

    def _constraints(self, props, component_name, par_name):

        if len(props) > 2:
            expr = ConstraintParser(props[2], self)
            expr.evaluate(component_name, par_name)

            import pdb
            pdb.set_trace()

            self.constraints += [expr.constraint]
        else:
            pass

    def parse_line(self, line):

        for par in self.par_names:
            props = line[par].split(',')
            self.p0 += [float(props[0])]
            self._bounds(props)

    def get_vars(self):

        d = {**vars(self), **self.fit_opts}
        todel = [
            'cfg', 'component_names', 'par_names', 'fit_opts', 'copts']
        for i in todel:
            del d[i]
        d['copts'] = self.copts
        d['constraints'] = self.constraints
        return d
