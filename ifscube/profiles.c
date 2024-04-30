#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double gauss(double x, double *p, int elements_in_p) {
    double g = 0.0;
    int number_of_parameters = 3;
    int number_of_gaussians = elements_in_p / number_of_parameters;
    int k;

    for (int i = 0; i < number_of_gaussians; ++i) {
        k = number_of_parameters * i;
        g += p[k] * exp(- pow(x - p[k + 1], 2.0) / (2.0 * pow(p[k + 2], 2.0)));
    }

    return g;
}
