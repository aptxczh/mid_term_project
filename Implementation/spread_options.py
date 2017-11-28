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
from stock_models import GBMModel, SVModel, ExpLevyModel


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
        self.r, self.T, self.S0, self.K = r, T, S0, K
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
        elif model == "SV":
            phi = SVModel(*args, **kwargs).phi
        elif model == "ExpLevy":
            phi = ExpLevyModel(*args, **kwargs).phi
        else:
            print("Illegal model input!")
            phi = lambda u: 0
        return np.exp(-self.r * self.T) * self.__payoff(N, eta, ep, phi)

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
            return gamma(1j * (u[0]+u[1]) - 1) * gamma(-1j * u[1]) / gamma(1j*u[0] + 1) * \
                   self.K**(-1j * (u[0]+u[1]) + 1)

        '''
        H_mat = np.empty((N, N), dtype=complex)
        for k1 in range(N):
            for k2 in range(N):
                u = -u_bar + np.array([k1, k2]) * eta + ep * 1j
                #if k1 == 1 and k2 == 1:
                #    print(phi(u))
                #    print(P_hat(u))
                H_mat[k1, k2] = -1**(k1+k2) * phi(u) * P_hat(u)

        res = (-1)**(l[0]+l[1]) * (eta * N)**2 * np.exp(-ep.dot(self.X0)) * ifft2(H_mat)[l[0], l[1]]
        '''
        sum = 0
        for k1 in range(N):
            for k2 in range(N):
                u = -u_bar + np.array([k1, k2]) * eta + ep * 1j
                sum += np.exp(2*np.pi*1j*np.array([k1, k2]).dot(l)/N) * (-1)**(k1+k2) * phi(u) * P_hat(u)
        res = (-1)**(l[0]+l[1]) * eta**2 * np.exp(-ep.dot(self.X0)) * sum

        return res.real

    def P(self, N, eta, ep):
        """
        payoff function for max(S1 - S2 - 1, 0). Used to test Theorem 1.1
        :param N: int, better be a power of 2. Number of lattice used to calculate FFT integral
        :param eta: int, FFT parameter
        :param ep: numpy array of length 2. FFT parameter
        :return: float
        """
        return self.__payoff(N, eta, ep, lambda u: 1)
