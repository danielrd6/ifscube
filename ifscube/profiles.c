#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>  // Include NumPy header


static PyObject *gauss_vel(PyObject *self, PyObject *args) {
    PyObject *wl_obj;  // Input Python object for the wavelength
    PyObject *rest_wl_obj;  // Input Python object for the wavelength
    PyObject *p_obj;  // Input Python object for the parameters
    PyArrayObject *wl_array;  // NumPy array for the wavelength
    PyArrayObject *rest_wl_array;  // NumPy array for the wavelength
    PyArrayObject *p_array;  // NumPy array for the parameters

    // Parse the input arguments and specify "O" for a generic Python object
    if (!PyArg_ParseTuple(args, "OOO", &wl_obj, &p_obj, &rest_wl_obj))
        return NULL;

    // Convert the input object to a NumPy array with any data type and flags
    wl_array = (PyArrayObject *)PyArray_FromAny(wl_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    rest_wl_array = (PyArrayObject *)PyArray_FromAny(rest_wl_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    p_array = (PyArrayObject *)PyArray_FromAny(p_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);

    if (wl_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert input to NumPy array");
        return NULL;
    }

    if (rest_wl_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert input to NumPy array");
        return NULL;
    }

    if (p_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert input to NumPy array");
        return NULL;
    }

    // Get pointers to the data in the NumPy array
    double *wavelength = (double *)PyArray_DATA(wl_array);
    double *rest_wavelength = (double *)PyArray_DATA(rest_wl_array);
    double *parameters = (double *)PyArray_DATA(p_array);
    npy_intp wl_size = PyArray_SIZE(wl_array);
    npy_intp rest_wl_size = PyArray_SIZE(rest_wl_array);
    npy_intp p_size = PyArray_SIZE(p_array);

    // Print array contents
    /*
    printf("Array contents:\n");
    for (npy_intp i = 0; i < wl_size; i++) {
        printf("%e ", wavelength[i]);
    }
    printf("\n");
    */

    double *flux = malloc(wl_size * sizeof(double));
    double lam_ratio, vel, w;
    const double c = 299792.458;
    int m = 3; // Number of parameters per component.
    int n;
    int n_functions = p_size / m; // Number of gaussians.

    for (int i = 0; i < wl_size; i++){
        flux[i] = 0.0;
    }

    for (int k = 0; k < p_size; k += m) {
        n = k / m;
        for (int i = 0; i < wl_size; i++) {
            lam_ratio = pow(wavelength[i] / rest_wavelength[n], 2.0);
            vel = c * (lam_ratio - 1.0) / (lam_ratio + 1.0);
            w = - pow((vel - parameters[k + 1]) / parameters[k + 2], 2.0) / 2.0;
            flux[i] += parameters[k] * exp(w) / (1.0 + (vel / c));
        }
    }

    // Create a NumPy array from the flux pointer
    PyObject *flux_array = PyArray_SimpleNewFromData(1, &wl_size, NPY_DOUBLE, flux);
    if (flux_array == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array from flux pointer");
        free(flux);
        Py_DECREF(wl_array);
        Py_DECREF(rest_wl_array);
        Py_DECREF(p_array);
        return NULL;
    }

    // Release the references to the input arrays
    Py_DECREF(wl_array);
    Py_DECREF(rest_wl_array);
    Py_DECREF(p_array);

    // Return the NumPy array (ownership transferred to the caller)
    return flux_array;
}

static PyMethodDef methods[] = {
    {"gauss_vel", gauss_vel, METH_VARARGS, "Gaussian in velocity space"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "elprofile",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_elprofile(void) {
    import_array();  // Initialize NumPy API
    return PyModule_Create(&moduledef);
}
