"""
fft.py

Implementation of numerical integration by FFT

Created: 11/26/17

Author: Bingcheng Wang, Yawei Wang & Zhihao Chen
"""

import numpy as np
from numpy.fft import ifft2

x1 = np.array([[1, 2], [3, 4]])
id = (0, 1)

print(x1[id])
