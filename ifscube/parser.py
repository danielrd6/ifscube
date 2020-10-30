# STDLIB
import configparser
import copy

import numpy as np

# LOCAL
from . import spectools


class ConstraintParser:

    def __init__(self, expr, parameter_names):

        self.parameter_names = parameter_names
        self.expr = expr
        self._operators = '+-*/><'
        self._containers = '()[]{}'
        self._spacer = ' \t\n\r'

        self._limits()
        self._classify()
        self._get_constraint_type()

        self.constraint = None

    @staticmethod
    def isnumber(s):

        try:
            float(s)
            return True
        except ValueError:
            return False

    def _what(self, c):

        if c.isalpha():
            group = 'alpha'
        elif c.isdecimal() or (c == '.'):
            group = 'number'
        elif c in self._operators:
            group = 'operator'
        elif c in self._spacer:
            group = 'spacer'
        else:
            group = 'unkonwn'

        return group

    def _limits(self):

        previous = ''
        lim = []

        for i, c in enumerate(self.expr):

            current = self._what(c)
            if current != previous:
                lim += [i]
                previous = copy.copy(current)
            else:
                continue

        self.lim = lim

    def _classify(self):

        operators = []
        variables = []
        containers = []
        numbers = []

        for c in self.expr.split(' '):

            if self.isnumber(c):
                numbers += [c]
            elif any([i.isalpha() for i in c]):
                variables += [c]
            elif self._what(c) == 'operator':
                operators += [c]
            elif self._what(c) == 'container':
                containers += [c]
            elif self._what(c) == 'unknown':
                continue

        self.numbers = numbers
        self.operators = operators
        self.variables = variables
        self.containers = containers

    def _idx(self, p_name):
        assert p_name in self.parameter_names, \
            f'Constraint expression "{self.expr}" refers to an undefined parameter "{p_name}".' \
            ' If you are using kinematic grouping, make sure your constraints refer only to the first feature' \
            ' of each group.'
        idx = self.parameter_names.index(p_name)
        return idx

    def _get_constraint_type(self):

        lis = self.tolist()

        if lis[0] in '<>':
            t = 'ineq'
        else:
            t = 'eq'

        self.type = t

    @staticmethod
    def _constraint_comparison(comparison, value):
        if comparison == '<':
            lb = -np.inf
            ub = value
        elif comparison == '>':
            lb = value
            ub = +np.inf
        else:
            lb = ub = value
        return lb, ub

    def _get_expression_elements(self, sentence):
        num = [i for i in sentence if i in self.numbers][0]
        var = [i for i in sentence if i in self.variables][0]
        idx2 = self._idx(var)
        a = float(num)
        oper = sentence[1]
        return num, var, idx2, a, oper

    def _differential_evolution_constraints(self, parameter, scale_factor):
        idx = self._idx(parameter)
        lis = self.tolist()
        m = np.zeros(2 * (len(self.parameter_names),))
        m[idx, idx] = 1.0

        if lis[0] in '<>':
            compare = lis[0]
            lis.remove(lis[0])
        else:
            compare = None

        if len(lis) == 1:
            if lis[0] in self.numbers:
                bounds = self._constraint_comparison(compare, float(lis[0]) / scale_factor)
            elif lis[0] in self.variables:
                m[idx, self._idx(lis[0])] = -1.0
                bounds = self._constraint_comparison(compare, 0.0)
            else:
                raise RuntimeError('Failed to interpret constraint expression.')
        elif len(lis) == 3:
            num, var, idx2, a, oper = self._get_expression_elements(lis)

            if oper == '+':
                m[idx, idx2] = -1.0
                bounds = self._constraint_comparison(compare, a / scale_factor)
            elif oper == '*':
                m[idx, idx2] = -a
                bounds = self._constraint_comparison(compare, 0.0)
            elif oper == '-':
                sign = 1.0 if (lis.index(num) > lis.index(var)) else -1.0
                m[idx, idx2] = -sign
                bounds = self._constraint_comparison(compare, sign * num)
            elif oper == '/':
                assert lis.index(num) > lis.index(var), 'Non linear constraints are not yet supported.'
                m[idx, idx2] = -(1.0 / a)
                bounds = self._constraint_comparison(compare, 0.0)
            else:
                raise RuntimeError('Failed to interpret constraint expression.')
        else:
            raise RuntimeError('Failed to interpret constraint expression.')

        lb = np.zeros(len(self.parameter_names))
        ub = np.zeros(len(self.parameter_names))
        lb[idx] = bounds[0]
        ub[idx] = bounds[1]
        self.constraint = {'matrix': m, 'lower_bounds': lb, 'upper_bounds': ub}

    def _minimize_constraints(self, parameter, scale_factor):
        idx = self._idx(parameter)
        lis = self.tolist()
        if lis[0] == '<':
            lis.remove(lis[0])
            sign = 1.
        elif lis[0] == '>':
            lis.remove(lis[0])
            sign = -1.
        else:
            sign = 1.

        if len(lis) == 1:
            if lis[0] in self.numbers:
                a = float(lis[0]) / scale_factor

                def func(x):
                    r = sign * (a - x[idx])
                    return r

            elif lis[0] in self.variables:
                idx2 = self._idx(lis[0])

                def func(x):
                    r = sign * (x[idx2] - x[idx])
                    return r
            else:
                raise Exception('Failed to interpret constraint expression.')
        elif len(lis) == 3:
            num, var, idx2, a, oper = self._get_expression_elements(lis)
            if lis.index(num) > lis.index(var):
                if oper == '+':
                    def op(x, j):
                        return x[j] + a
                elif oper == '-':
                    def op(x, j):
                        return x[j] - a
                elif oper == '/':
                    def op(x, j):
                        return x[j] / a
                elif oper == '*':
                    def op(x, j):
                        return x[j] * a
            else:
                if oper == '+':
                    def op(x, j):
                        return a + x[j]
                elif oper == '-':
                    def op(x, j):
                        return a - x[j]
                elif oper == '/':
                    def op(x, j):
                        return a / x[j]
                elif oper == '*':
                    def op(x, j):
                        return a * x[j]

            def func(x):
                r = sign * (op(x, idx2) - x[idx])
                return r

        else:
            raise Exception('Failed to interpret constraint expression.')
        self.constraint = dict(type=self.type, fun=func)

    def evaluate(self, parameter, scale_factor: float = 1.0, method: str = 'slsqp'):
        if method == 'differential_evolution':
            self._differential_evolution_constraints(parameter, scale_factor)
        elif method in ['slsqp', 'trust-constr']:
            self._minimize_constraints(parameter, scale_factor)
        else:
            raise RuntimeError(f'Unknown method "{method}".')

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

        # return elements
        elements = self.expr.split(' ')
        elements = [i for i in elements if i != '']
        return elements


class LineFitParser:

    def __init__(self, *args, **kwargs):

        self.component_names = []
        self.bounds = []
        self.constraints = []
        self.cwin = {}
        self.features = []

        if args or kwargs:
            self._load(*args, **kwargs)

    def _load(self, config_file):

        if isinstance(config_file, str):
            self.cfg = configparser.ConfigParser()
            self.cfg.read(config_file)
        else:
            self.cfg = config_file

        par_names = {
            'gaussian': ('rest_wavelength', 'amplitude', 'velocity', 'sigma', 'k_group'),
            'gauss_hermite': ('rest_wavelength', 'amplitude', 'velocity', 'sigma', 'h3', 'h4', 'k_group'),
        }

        self.par_names = par_names[self.cfg.get('fit', 'function', fallback='gaussian')]
        self.component_names = [
            i for i in self.cfg.sections() if i not in [
                'continuum', 'minimization', 'fit', 'equivalent_width',
                'loading']]

        # Each section has to be a line, except for the DEFAULT, MINOPTS,
        # and CONTINUUM sections, and equivalent_width.
        for line in self.component_names:
            self.parse_line(line)

        unconstrained_parameters = ['rest_wavelength', 'k_group', 'continuum_windows']
        for line in self.component_names:
            for par in [_ for _ in self.cfg.options(section=line) if _ not in unconstrained_parameters]:
                prop = self.cfg[line][par].split(',')
                self._constraints(prop, f'{line}.{par}')

        self._fit()
        # self._kinematic_constraints()
        self._minimization()
        self._continuum()
        self._eqw()
        self._loading()

    def _idx(self, cname, pname):

        ci = self.component_names.index(cname)
        pi = self.par_names.index(pname)
        npars = len(self.par_names)

        idx = ci * npars + pi

        return idx

    def _parse_dict(self, section, float_args=(), int_args=(), bool_args=()):

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

        if 'minimization' in self.cfg.sections():
            self.minopts = self._parse_dict(
                section='minimization',
                float_args=['eps', 'ftol'],
                int_args=['maxiter'],
                bool_args=['disp'],
            )
        else:
            self.minopts = {}

    def _fit(self):

        if 'fit' not in self.cfg.sections():
            self.fit_opts = {}
            return

        fit_opts = {**self.cfg['fit']}
        string_values = ['peak', 'cofm']

        key = 'individual_spec'
        if key in fit_opts:
            if fit_opts[key] in string_values:
                pass
            else:
                try:
                    fit_opts[key] = self.cfg.getboolean('fit', key)
                    assert fit_opts[key] is not True, \
                        '*individual_spec* must be "peak", "cofm", "no" or a' \
                        'pair of spaxel coordinates "x, y".'
                except ValueError:
                    fit_opts[key] = tuple([int(i) for i in fit_opts[key].split(',')])

        key = 'spiral_center'
        if key in fit_opts:
            if fit_opts[key] in string_values:
                pass
            else:
                fit_opts[key] = tuple([int(i) for i in fit_opts[key].split(',')])

        boolean_args = ['write_fits', 'refit', 'update_bounds', 'spiral_loop',
                        'verbose', 'fit_continuum', 'optimize_fit',
                        'guess_parameters', 'trivial', 'test_jacobian', 'debug',
                        'intrinsic_sigma_constr']
        for i in boolean_args:
            if i in fit_opts:
                fit_opts[i] = self.cfg.getboolean('fit', i)

        fit_opts['fixed'] = self.cfg.getboolean('fit', 'fixed', fallback=False)

        float_args = ['refit_radius', 'sig_threshold', 'instrument_dispersion', 'optimization_window',
                      'good_minfraction', 'instrument_dispersion_angstrom']
        for i in float_args:
            if i in fit_opts:
                fit_opts[i] = self.cfg.getfloat('fit', i)

        fit_opts['monte_carlo'] = self.cfg.getint('fit', 'monte_carlo', fallback=0)
        fit_opts['method'] = self.cfg.get('fit', 'method', fallback='slsqp')
        fit_opts['suffix'] = self.cfg.get('fit', 'suffix', fallback='_linefit')

        bounds_change = self.cfg.get('fit', 'bounds_change', fallback=None)
        if bounds_change is not None:
            fit_opts['bounds_change'] = [float(_) for _ in bounds_change.split(',')]
        else:
            fit_opts['bounds_change'] = bounds_change

        key = 'fitting_window'
        if key in fit_opts:
            a = fit_opts[key].split(':')
            fit_opts['fitting_window'] = tuple([float(i) for i in a])

        self.fit_opts = fit_opts

    def _loading(self):

        if 'loading' in self.cfg.sections():
            self.loading_opts = self._parse_dict(section='loading', float_args=('redshift',), int_args=('wcs_axis',))
        else:
            self.loading_opts = {}

        fits_extension_keys = list(self.loading_opts.keys())
        if 'redshift' in fits_extension_keys:
            fits_extension_keys.remove('redshift')

        for key in fits_extension_keys:
            if self.loading_opts[key] == 'None':
                self.loading_opts[key] = None
            else:
                # NOTE: MEF extension specifications might be integers.
                try:
                    self.loading_opts[key] = int(self.loading_opts[key])
                except ValueError:
                    pass

        return

    def _eqw(self):

        if 'equivalent_width' in self.cfg.sections():
            self.eqw_opts = self._parse_dict(
                section='equivalent_width',
                float_args=('sigma_factor',))
        else:
            self.eqw_opts = {}

        self.eqw_opts['continuum_windows'] = self.cwin

    def _continuum(self):

        if 'continuum' in self.cfg.sections():
            self.copts = self._parse_dict(
                section='continuum',
                float_args=['lower_threshold', 'upper_threshold', 'line_weight'],
                int_args=['degree', 'n_iterate'],
                bool_args=[],
            )
        else:
            self.copts = {}

    def _continuum_windows(self, line, line_pars):

        if 'continuum_windows' in line_pars:
            self.cwin[line] = [
                float(i) for i in
                line_pars.get('continuum_windows').split(',')]
        else:
            self.cwin[line] = None

    def _bounds(self, line, par, props):

        if len(props) > 1:
            if (~props[1].isspace()) and (props[1] != ''):
                if '+-' in props[1]:
                    number = float(props[1].split()[-1])
                    low, up = [float(props[0]) + i for i in [-number, +number]]
                elif ':' in props[1]:
                    low, up = [float(i) if (i != '') else None for i in props[1].split(':')]
                else:
                    raise Exception('Failed to parse bounds.')
                b = (low, up)
            else:
                b = (None, None)
            self.bounds += [[line, par, b]]
        else:
            b = (None, None)

    def _constraints(self, props, par_name):

        if len(props) > 2:
            if (~props[2].isspace()) and (props[2] != ''):
                # expr = ConstraintParser(props[2], parameter_names=self.par_names)
                # expr.evaluate(par_name)

                # self.constraints += [expr.constraint]
                self.constraints += [(par_name, props[2])]

    def _kinematic_constraints(self):

        cn = self.k_component_names
        kg = self.k_groups

        # Fix instrinsic velocity dispersion (sigma0) parameters 
        fix_sigma0 = self.fit_opts.get('intrinsic_sigma_constr', False)
        instdisp = self.fit_opts.get('instrument_dispersion_angstrom', 0)
        # Remove key because it is not an argument of the linefit function
        self.fit_opts.pop('intrinsic_sigma_constr', None)
        self.fit_opts.pop('instrument_dispersion_angstrom', None)

        if fix_sigma0:
            instdisp_vel = instdisp * 299792.458 / np.array(self.feature_wl)

        components = []
        for i in set(kg):
            components += [[cn[j] for j in range(len(cn)) if i == kg[j]]]

        for i, g in enumerate(components):
            if len(g) > 1:
                for j in range(len(g[:-1])):
                    for m in ['velocity', 'sigma', 'h3', 'h4']:
                        if m in self.par_names:
                            par1 = self._idx(g[j], m)
                            par2 = self._idx(g[j + 1], m)

                            if (m == 'sigma') & fix_sigma0:
                                self.constraints += [
                                    spectools.Constraints.same_intrinsic_sigma(par1, par2,
                                                                               instdisp_vel[j] ** 2 - instdisp_vel[
                                                                                   j + 1] ** 2)]
                            else:
                                self.constraints += [
                                    spectools.Constraints.same(par1, par2)]

        self.k_components = components

    def parse_line(self, line):

        d = {'name': line}
        line_pars = self.cfg[line]
        for par in [_ for _ in self.cfg.options(section=line)]:
            if par == 'continuum_windows':
                pass
            elif par == 'k_group':
                d.update({'kinematic_group': line_pars.getint(par)})
            elif par == 'rest_wavelength':
                d.update({par: line_pars.getfloat(par)})
            elif par == 'fixed':
                d.update({par: line_pars.getboolean(par)})
            else:
                props = line_pars[par].split(',')
                if props[0] not in ['peak', 'mean', 'median']:
                    x = float(props[0])
                else:
                    x = props[0]

                d.update({par: x})
                self._bounds(line, par, props)

        self.features += [d]

    def get_vars(self):

        d = {**vars(self), **self.fit_opts}
        todel = ['cfg', 'par_names', 'fit_opts', 'copts', 'loading_opts', 'cwin']
        for i in todel:
            del d[i]
        d['copts'] = self.copts
        d['constraints'] = self.constraints
        return d
