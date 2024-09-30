import numpy as np
from scipy.stats import norm
from enum import Enum

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class Greeks:
    CALL = "call"
    PUT = "put"
    
    def __init__(self, S: float, K: float, T: float, r: float, q: float, sigma: float,
                 option_type: OptionType):
        """
        Initialize the class with common option parameters.

        Parameters:
        S : float : initial stock price
        K : float : strike price
        T : float : time to expiration (in years)
        r : float : risk-free rate
        q : float : dividend yield
        sigma : float : volatility
        option_type : str : 'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.option_type = option_type
        
    def _d1_d2(self):
            """Calculate d1 and d2 used in the Black-Scholes model."""
            d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
            d2 = d1 - self.sigma * np.sqrt(self.T)
            return d1, d2

    def delta(self) -> float:
            """Calculate the delta of the option."""
            N = norm.cdf
            d1, _ = self._d1_d2()
            
            if self.option_type == self.CALL:
                return N(d1)
            elif self.option_type == self.PUT:
                return N(d1) - 1

    def gamma(self) -> float:
            """Calculate the gamma of the option."""
            n = norm.pdf
            d1, _ = self._d1_d2()
            return n(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
            """Calculate the vega of the option."""
            n = norm.pdf
            d1, _ = self._d1_d2()
            return self.S * n(d1) * np.sqrt(self.T)

    def theta(self) -> float:
            """Calculate the theta of the option."""
            N = norm.cdf
            n = norm.pdf
            d1, d2 = self._d1_d2()
            
            if self.option_type == self.CALL:
                return (-self.S * n(d1) * self.sigma / (2 * np.sqrt(self.T))
                        - self.r * self.K * np.exp(-self.r * self.T) * N(d2))
            elif self.option_type == self.PUT:
                return (-self.S * n(d1) * self.sigma / (2 * np.sqrt(self.T))
                        + self.r * self.K * np.exp(-self.r * self.T) * N(-d2))

    def rho(self) -> float:
            """Calculate the rho of the option."""
            N = norm.cdf
            _, d2 = self._d1_d2()
            
            if self.option_type == self.CALL:
                return self.K * self.T * np.exp(-self.r * self.T) * N(d2)
            elif self.option_type == self.PUT:
                return -self.K * self.T * np.exp(-self.r * self.T) * N(-d2)