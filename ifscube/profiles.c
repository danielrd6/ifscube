#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>  // Include NumPy header

static PyObject *gauss_vel(PyObject *self, PyObject *args) {
    PyObject *wl_obj;  // Input Python object for the wavelength
    PyObject *p_obj;  // Input Python object for the parameters
    PyArrayObject *wl_array;  // NumPy array for the wavelength
    PyArrayObject *p_array;  // NumPy array for the parameters

    // Parse the input arguments and specify "O" for a generic Python object
    if (!PyArg_ParseTuple(args, "OO", &wl_obj, &p_obj))
        return NULL;

    // Convert the input object to a NumPy array with any data type and flags
    wl_array = (PyArrayObject *)PyArray_FromAny(wl_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    p_array = (PyArrayObject *)PyArray_FromAny(p_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);

    if (wl_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert input to NumPy array");
        return NULL;
    }

    if (p_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert input to NumPy array");
        return NULL;
    }

    // Get pointers to the data in the NumPy array
    double *wavelength = (double *)PyArray_DATA(wl_array);
    double *parameters = (double *)PyArray_DATA(p_array);
    npy_intp wl_size = PyArray_SIZE(wl_array);
    npy_intp p_size = PyArray_SIZE(p_array);

    // Print array contents
    printf("Array contents:\n");
    for (npy_intp i = 0; i < wl_size; i++) {
        printf("%e ", wavelength[i]);
    }
    printf("\n");

    double velocity[wl_size];
    for (int i = 0; i < wl_size; i++) {
        velocity[i] = wavelength[i] * 2.0;
    }

    // Release the reference to the input array
    Py_DECREF(wl_array);
    Py_DECREF(p_array);

    // Return a result (if needed)
    // For now, return None
    Py_INCREF(Py_None);
    return Py_None;
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
