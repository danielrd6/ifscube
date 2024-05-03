#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>  // Include NumPy header

static PyObject *gauss_vel(PyObject *self, PyObject *args) {
    PyObject *wl_obj;  // Input Python object for the wavelength vector
    PyObject *rest_wl_obj;  // Input Python object for the rest wavelength of spectral features
    PyObject *p_obj;  // Input Python object for the function parameters
    PyArrayObject *wl_array;  // NumPy array for the wavelength
    PyArrayObject *rest_wl_array;  // NumPy array for the rest wavelength
    PyArrayObject *p_array;  // NumPy array for the parameters

    // Parse the input arguments and specify "O" for a generic Python object
    if (!PyArg_ParseTuple(args, "OOO", &wl_obj, &rest_wl_obj, &p_obj))
        return NULL;

    // Convert the input object to a NumPy array with any data type and flags
    wl_array = (PyArrayObject *)PyArray_FromAny(wl_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    rest_wl_array = (PyArrayObject *)PyArray_FromAny(rest_wl_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    p_array = (PyArrayObject *)PyArray_FromAny(p_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);

    // Get pointers to the data in the NumPy array
    double *wavelength = (double *)PyArray_DATA(wl_array);
    double *rest_wavelength = (double *)PyArray_DATA(rest_wl_array);
    double *parameters = (double *)PyArray_DATA(p_array);
    npy_intp wl_size = PyArray_SIZE(wl_array);
    npy_intp rest_wl_size = PyArray_SIZE(rest_wl_array);
    npy_intp p_size = PyArray_SIZE(p_array);

    double *flux = malloc(wl_size * sizeof(double));
    double lam_ratio, vel, w;
    const double c = 299792.458;
    int m = 3; // Number of parameters per gaussian.
    int n;
    int n_functions = p_size / m; // Number of gaussians.

    memset(flux, 0.0, wl_size * sizeof(double));

    for (int k = 0; k < p_size; k += m) {
        n = k / m;
        for (int i = 0; i < wl_size; i++) {
            // lam_ratio is the square of lambda_received divided by lambda_source.
            lam_ratio = (wavelength[i] * wavelength[i]) / (rest_wavelength[n] * rest_wavelength[n]);
            // vel is the velocity
            vel = c * (lam_ratio - 1.0) / (lam_ratio + 1.0);
            // w is the square of ((velocity - velocity_0) / sigma)
            w = ((vel - parameters[k + 1]) / parameters[k + 2]) * ((vel - parameters[k + 1]) / parameters[k + 2]);
            flux[i] += parameters[k] * exp(- w / 2.0) / (1.0 + (vel / c));
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
};

static PyMethodDef methods[] = {
    {"gauss_vel", gauss_vel, METH_VARARGS, "Gaussians in velocity space."},
    // {"gausshermite_vel", gauss_vel, METH_VARARGS, "Gaussians in velocity space."},
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
};
