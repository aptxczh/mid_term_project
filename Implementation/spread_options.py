"""
spread_options.py

Class of spread option. Priced by FFT method.

Created: 11/26/17

Author: Bingcheng Wang, Yawei Wang & Zhihao Chen
"""

import numpy as np
#from mpmath import gamma
from scipy.special import gamma
from numpy.fft import ifft2
from stock_models import GBMModel


class SpreadOption(object):
    """
    spread option, its price, error and greeks
    """
    def __init__(self, S0, K, T, r):
        """
        constructor
        :param S0: numpy array of length 2, initial prices of underlying asset
        :param K: float, strike
        :param T: float, time to maturity
        :param r: float, constant interest rate
        """
        S0 = np.array(S0)
        assert len(S0) == 2, "S0 should be a vector of length 2"
        self.r, self.T, self.S0, self.K = r, T, S0 / K, K   # scaled S
        self.X0 = np.log(S0)

    def price(self, N, eta, ep, model="GBM", *args, **kwargs):
        """
        get the price of the spread option
        :param N: int, better be a power of 2. Number of lattice used to calculate FFT integral
        :param eta: int, FFT parameter
        :param ep: numpy array of length 2. FFT parameter
        :param model: can be either "GBM", "SV", "ExpLevy"
        :param args: other parameters passed to characteristic function phi
        :param kwargs: other parameters passed to characteristic function phi
        :return: float
        """
        ep = np.array(ep)
        assert len(ep) == 2, "ep should be a vector of length 2"
        assert ep[1] > 0, "ep2 > 0"
        assert ep[0] + ep[1] < -1, "ep1 + ep2 < -1"

        if model == "GBM":
            phi = GBMModel(*args, **kwargs).phi
        else:
            phi = lambda u: 0
        return np.exp(-self.r * self.T) * self.__payoff(N, eta, ep, phi) * self.K

    def __payoff(self, N, eta, ep, phi):
        return 1 / (4 * np.pi**2) * self.__double_int(N, eta, ep, phi)

    def __double_int(self, N, eta, ep, phi):
        # use FFT to calculate the double integral
        u_bar = N * eta / 2
        eta_star = np.pi / u_bar
        x_bar = N * eta_star / 2
        l = (self.X0 + x_bar) / eta_star

        l = l.astype(int)  # convert to int

        def P_hat(u):
            return gamma(1j * (u[0]+u[1]) - 1) * gamma(-1j * u[1]) / gamma(1j*u[0] + 1)

        H_mat = np.empty((N, N), dtype=complex)
        for k1 in range(N):
            for k2 in range(N):
                u = -u_bar + np.array([k1, k2]) * eta + ep * 1j
                #if k1 == 1 and k2 == 1:
                #    print(phi(u))
                #    print(P_hat(u))
                H_mat[k1, k2] = -1**(k1+k2) * phi(u) * P_hat(u)

        res = (-1)**(l[0]+l[1]) * (eta * N)**2 * np.exp(-ep.dot(self.X0)) * ifft2(H_mat)[l[0], l[1]]

        return res

    def P(self, N, eta, ep):
        return self.__payoff(N, eta, ep, lambda u: 1)


## Parameters
r = 0.1
T = 1.0
rho = 0.5
delta_1 = 0.05
sigma_1 = 0.2
delta_2 = 0.05
sigma_2 = 0.1

S0 = np.array([10, 5])
N = 64
u_bar = 20
eta = u_bar * 2 / N
ep = np.array([-4.2, 2.1])

p = SpreadOption(S0, 1, T, r).P(N, eta, ep)

print(max((S0[0] - S0[1] - 1), 0))
print(p)
