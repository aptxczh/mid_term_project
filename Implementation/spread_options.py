"""
spread_options.py

Class of spread option. Priced by FFT method.

Created: 11/26/17

Author: Bingcheng Wang, Yawei Wang & Zhihao Chen
"""

import numpy as np

class SpreadOption(object):
    """
    spread option, its price, error and greeks
    """
    def __init__(self, S0, K, T, r):
        """
        constructor
        :param r: float, constant interest rate
        :param T: float, time to maturity
        :param S0: numpy array with length 2, initial prices of underlying asset
        :param K: float, strike
        """
        S0 = np.array(S0)
        assert len(S0) == 2, "S0 should be a vector of length 2"
        self.r, self.T, self.S0, self.K = r, T, S0 / K, K   # scaled S
        self.X0 = np.log(S0)

    def price(self):
        return np.exp(-self.r * self.T) * self._payoff() * self.K

    def _payoff(self, *args, **kwargs):
        return 1 / (4 * np.pi**2) * self._double_int(*args, **kwargs)

    def _double_int(self, ):
        return 0