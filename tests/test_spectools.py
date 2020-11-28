import numpy as np

from ifscube import spectools


def test_find_intermediary_value():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([10, 20, 30, 40, 50])
    xv = spectools.find_intermediary_value(x, y, 31)
    assert xv == 2.1
