"""
stock_models.py

Classes for three kinds of stock models listed on the paper.

Created: 11/26/17

Author: Bingcheng Wang, Yawei Wang & Zhihao Chen
"""

import numpy as np


class GBMModel(object):
    """.
    Two-asset Black-Scholes model
    """
    def __init__(self, sigma, r, T, rho):
        """
        constructor
        :param sigma: numpy array with length 2, constant volatility
        :param r: float, constant interest rate
        :param T: float, time to maturity
        :param rho: float, constant correlation between two Brownian motion, |rho|<1
        """
        sigma = np.array(sigma)
        assert len(sigma) == 2
        self.sigma, self.r, self.T, self.rho = sigma, r, T, rho

    def phi(self, u):
        """
        joint characteristic function, as a function of u
        :param u: numpy array with length 2
        """
        e = np.matrix([1, 1])
        sigma_ = np.matrix([[self.sigma[0]**2, self.rho*self.sigma[0]*self.sigma[1]],
                           [self.rho*self.sigma[0]*self.sigma[1], self.sigma[1]**2]])
        return np.exp(u*(self.r*self.T*e-self.sigma**2*self.T/2).T*1j-u*sigma_*np.reshape(u, (2, 1))*self.T/2)


class SVModel(object):
    """
    Two-asset Three factor SV model
    """
    def __init__(self,sigma,r,T,rho,kappa,mu,v0,delta):
        """
        constructor
        :param sigma: numpy array with length 3, constant volatility
        :param r: float, constant interest rate
        :param T: float, time to maturity
        :param rho: numpy array with length 3, constant correlation between three Brownian motion
        :param kappa: float, mean revertion speed
        :param mu:float, long term volatility
        :param v0:float, initial volatility
        :param delta:numpy array with length 2
        """
        sigma = np.array(sigma)
        rho = np.array(rho)
        assert len(sigma) == 2
        assert len(rho) == 3
        self.sigma, self.r, self.T, self.rho, self.kappa, self.mu, self.v0, self.delta \
            = sigma, r, T, rho, kappa, mu, v0, delta

    def phi(self, u):
        """
        joint characteristic function, as a function of u
        :param u: numpy array with length 2
        """
        zeta = -(self.sigma[0]**2*u[0]**2+self.sigma[1]**2*u[1]**2+2*self.rho[0]*self.sigma[0]*self.sigma[1]*u[0]*u[1] +
               (self.sigma[0]**2*u[0]+self.sigma[1]**2*u[1])*1j)/2
        gamma = self.kappa - (self.rho[1]*self.sigma[0]*u[0]+self.rho[2]*self.sigma[1]*u[1])*self.sigma[2]*1j
        theta = np.sqrt(gamma**2-2*self.sigma[2]**2*zeta)
        e = np.matrix([1, 1])

        return (2*zeta*(1-np.exp(-theta*self.T))/(2*theta-(theta-gamma)*(1-np.exp(-theta*self.T))))*self.v0\
               + u*(self.r*e-self.delta).T*1j - \
               self.kappa*self.mu*(2*np.log((2 * theta-(theta-gamma)*(1-np.exp(-theta*self.T)))/2/theta)+
                                   (theta-gamma)*self.T)/self.sigma[2]**2


class ExpLevyModel(object):
    """
    Two-asset Exponential Levy model
    """
    def __init__(self):
        pass

    def phi(self):
        pass
