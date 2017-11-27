"""
spread_options.py

Class of spread option. Priced by FFT method.

Created: 11/26/17

Author: Bingcheng Wang, Yawei Wang & Zhihao Chen
"""

import numpy as np
from scipy.special import gamma
from numpy.fft import ifft2


class SpreadOption(object):
    """
    spread option, its price, error and greeks
    """
    def __init__(self, S0, K, T, r):
        """
        constructor
        :param S0: numpy array with length 2, initial prices of underlying asset
        :param K: float, strike
        :param T: float, time to maturity
        :param r: float, constant interest rate
        """
        S0 = np.array(S0)
        assert len(S0) == 2, "S0 should be a vector of length 2"
        self.r, self.T, self.S0, self.K = r, T, S0 / K, K   # scaled S
        self.X0 = np.log(S0)

    def price(self, N, eta, ep, method="GBM", *args, **kwargs):
        if method == "GBM":
            phi = lambda x: 0  # TODO
        else:
            phi = lambda x: 0  # TODO
        return np.exp(-self.r * self.T) * self.__payoff(N, eta, ep, phi, *args, **kwargs) * self.K

    def __payoff(self, *args, **kwargs):
        return 1 / (4 * np.pi**2) * self.__double_int(*args, **kwargs)

    def __double_int(self, N, eta, ep, phi, *args, **kwargs):
        ep = np.array(ep)
        assert len(ep) == 2, "ep should be a vector of length 2"
        assert ep[1] > 0, "ep2 > 0"
        assert ep[0] + ep[1] < -1, "ep1 + ep2 < -1"
        # use FFT to calculate the double integral
        u_bar = N * eta / 2
        eta_star = np.pi / u_bar
        x_bar = N * eta_star / 2
        l = (self.X0 + x_bar) / eta_star

        if not np.array_equal(np.floor(l), l):
            print("l is not an array of integer!")

        l = l.astype(int)  # convert to int

        def P_hat(k):
            u = -u_bar + k * eta + ep * 1j
            return gamma(1j * (u[0]+u[1]) - 1) * gamma(-1j * u[1]) / gamma(1j*u[0] + 1)

        H_mat = np.empty((N, N))
        for k1 in range(N):
            for k2 in range(N):
                H_mat[k1, k2] = -1**(k1+k2) * phi(*args, **kwargs) * P_hat(np.array([k1, k2]))
        res = (-1)**(l[0]+l[1]) * (eta * N)**2 * np.exp(-ep.dot(self.X0.T)) * ifft2(H_mat)[l]

        return res
