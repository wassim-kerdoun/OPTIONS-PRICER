import numpy as np
from scipy.optimize import least_squares
from options.EUop import EUop
from enum import Enum

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class ImpliedVolatility:
    CALL = "call"
    PUT = "put"
    
    def __init__(self, S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType):
        """
        Initialize the ImpliedVolatility class.

        Parameters:
        - S: Current stock price.
        - K: Strike price.
        - T: Time to expiration (in years).
        - r: Risk-free interest rate (as a decimal).
        - q: Dividend yield (as a decimal).
        - sigma: Initial volatility estimate (as a decimal).
        - option_type: 'call' or 'put'.
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.option_type = option_type

    def implied_volatility_least_squares(self, market_price: float) -> float:
        """
        Calculate the implied volatility using the least squares method.

        Parameters:
        - market_price: Market price of the option.

        Returns:
        - Implied volatility (as a decimal).
        """
        def objective_function(sigma):
            return EUop(self.S, self.K, self.T, self.r, self.q, sigma, self.option_type).black_scholes() - market_price

        result = least_squares(objective_function, x0=[0.2])
        return result.x[0]
    
    def implied_volatility_newton_raphson(self, market_price: float, 
                                            initial_guess: float = 0.25, max_iterations: int = 500, 
                                            tolerance: float = 1e-4) -> float:
        """
        Calculate the implied volatility using the Newton-Raphson method.

        Parameters:
        - market_price: Market price of the option.
        - initial_guess: Initial guess for volatility.
        - max_iterations: Maximum number of iterations for convergence.
        - tolerance: Convergence tolerance for the price difference.

        Returns:
        - Implied volatility (as a decimal).

        Raises:
        - ValueError: If convergence is not achieved.
        """
        sigma = initial_guess
        for _ in range(max_iterations):
            price = EUop(self.S, self.K, self.T, self.r, self.q, sigma, self.option_type).black_scholes()
            vega_value = EUop(self.S, self.K, self.T, self.r, self.q, sigma, self.option_type).vega()
            
            price_diff = price - market_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            sigma -= price_diff / vega_value
            
        raise ValueError("Convergence not achieved in Newton-Raphson method.")

    def volatility_smile(self):
        """
        Compute the volatility smile for a range of strike prices.

        Parameters:
        None

        Returns:
        - A range of generated Strike (Ks) and computed implied volatility (implied_vols).
        
        """
        Ks = np.linspace(self.K - 50, self.K + 50, 100)  # Standard range for strike prices
        implied_vols = []

        for K in Ks:
            calculated_market_price = EUop(self.S, K, self.T, self.r, self.q, self.sigma, self.option_type).black_scholes()
            implied_vol = self.implied_volatility_least_squares(calculated_market_price)
            implied_vols.append(implied_vol)

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=Ks / self.S, y=implied_vols, mode='lines+markers', name='Implied Volatility'))
        # fig.update_layout(
        #     xaxis_title="Moneyness (K/S)",
        #     yaxis_title="Implied Volatility",
        #     title=f"Volatility Smile ({self.option_type.capitalize()} Option)",
        #     xaxis=dict(showgrid=True),
        #     yaxis=dict(showgrid=True)
        # )
        # fig.show()
        
        return Ks, implied_vols

    def volatility_term_structure(self):
        """
        Compute the volatility term structure for a range of times to expiration.

        Parameters:
        None

        Returns:
        - A range of generated times to expiration (T_values) and computed implied volatility (implied_vols).
        """
        
        T_values = np.linspace(1/12, 2.0, num=48)
        implied_vols = []

        for T in T_values:
            calculated_market_price = EUop(self.S, self.K, T, self.r, self.q, self.sigma, self.option_type).black_scholes()
            implied_vol = self.implied_volatility_least_squares(calculated_market_price)
            implied_vols.append(implied_vol)

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=T_values, y=implied_vols, mode='lines+markers', name='Implied Volatility'))
        # fig.update_layout(
        #     xaxis_title="Time to Expiration (Years)",
        #     yaxis_title="Implied Volatility",
        #     title=f"Implied Volatility Term Structure ({self.option_type.capitalize()} Option)",
        #     xaxis=dict(showgrid=True),
        #     yaxis=dict(showgrid=True)
        # )
        # fig.show()
        
        return T_values, implied_vols

    def volatility_surface(self):
        """
        Compute the volatility surface for a range of strike prices and times to expiration.

        Parameters:
        None

        Returns:
        - A range of generated strike prices (K_values), times to expiration (T_values) 
        and computed implied volatility surface (implied_vol_matrix).
        """
        
        K_values = np.linspace(self.K - self.K*0.05, self.K + self.K*0.05, 48)
        T_values = np.linspace(1/12, 2.0, num=48)
        implied_vol_matrix = np.zeros((len(T_values), len(K_values)))

        for i, T in enumerate(T_values):
            for j, K in enumerate(K_values):
                calculated_market_price = EUop(self.S, K, T, self.r, self.q, self.sigma, self.option_type).black_scholes()
                implied_vol_matrix[i, j] = self.implied_volatility_least_squares(calculated_market_price)

        # fig = go.Figure(data=[go.Surface(x=K_values / self.S, y=T_values, z=implied_vol_matrix, colorscale='Viridis', showscale=True)])

        # fig.update_layout(
        #     title=f'Implied Volatility Surface ({self.option_type.capitalize()} Options)',
        #     scene=dict(
        #         xaxis_title='Moneyness (K/S)',
        #         yaxis_title='Time to Expiration (Years)',
        #         zaxis_title='Implied Volatility'
        #     )
        # )

        # fig.show()
        
        return K_values, T_values, implied_vol_matrix
