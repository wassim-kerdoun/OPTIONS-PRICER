import numpy as np
import math
from scipy.stats import norm
from enum import Enum
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO


class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class EUop:
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

    def black_scholes(self) -> float:
        """
        Calculate the price of European options using the Black-Scholes model.
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == self.CALL:
            return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == self.PUT:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'.")

    def merton_jump_diffusion(self, mu_j: float, sigma_j: float, lam: float, 
                              max_iter: int = 100, stop_cond: float = 1e-15) -> float:
        """
        Calculate the price of European options using the Merton jump diffusion model.
        """
        V = 0
        
        for k in range(max_iter):
            sigma_k = np.sqrt(self.sigma**2 + k * sigma_j**2 / self.T)
            r_k = self.r - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1) + (k * (mu_j + 0.5 * sigma_j**2)) / self.T
            poisson_weight = (np.exp(-lam * self.T) * (lam * self.T)**k / math.factorial(k))
            
            bs_value = self.black_scholes()
            sum_k = poisson_weight * bs_value
            V += sum_k
            
            if sum_k < stop_cond:  # if the last added component is below the threshold, return the current sum
                return V
        
        return V
    
    def european_binomial(self, n: int) -> tuple:
        """
        Price a European option using the binomial tree model.
        """
        dt = self.T / n
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        if not 0 <= p <= 1:
            raise ValueError("Arbitrage detected: check inputs for no-arbitrage conditions.")

        asset_prices = np.zeros((n + 1, n + 1))
        option_prices = np.zeros((n + 1, n + 1))

        for j in range(n + 1):
            asset_prices[j][n] = self.S * (u ** (n - j)) * (d ** j)
            print(f"Asset Price at node ({j}, {n}): {asset_prices[j][n]}")
            
        for j in range(n + 1):
            if self.option_type.lower() == 'call':
                option_prices[j][n] = max(0, asset_prices[j][n] - self.K)
            elif self.option_type.lower() == 'put':
                option_prices[j][n] = max(0, self.K - asset_prices[j][n])

        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                asset_prices[j][i] = self.S * (u ** (i - j)) * (d ** j)
                print(f"Payoff for Option at node ({j}, {n}): {option_prices[j][n]}")
                option_prices[j][i] = np.exp(-self.r * dt) * (p * option_prices[j][i + 1] + (1 - p) * option_prices[j + 1][i + 1])

        return option_prices, asset_prices, u, d, p, dt

    def draw_binomial_tree(self, stock_tree, option_tree, n, u, d, p, dt):
        G = nx.DiGraph()

        for i in range(n + 1):
            for j in range(i + 1):
                stock_price = stock_tree[j][i]
                if i == n:
                    payoff = option_tree[j][n]
                    label = f"Asset: \n${stock_price:.2f}\nPayoff: \n${payoff:.2f}"
                    node_color = 'salmon'
                else:
                    option_price = option_tree[j][i]
                    label = f"Asset: \n${stock_price:.2f}\nOption: \n${option_price:.2f}"
                    node_color = 'lightblue'

                G.add_node((i, j), label=label, color=node_color)

                if i < n:
                    G.add_edge((i, j), (i + 1, j + 1))  # Up move
                    G.add_edge((i, j), (i + 1, j))      # Middle (stay)

        pos = {(x, y): (x * dt, -y) for x, y in G.nodes()}

        plt.figure(figsize=(12, 8))
        
        colors = [G.nodes[node]['color'] for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=3500, node_color=colors, font_size=10,
                labels=nx.get_node_attributes(G, 'label'))

        plt.axis('off')

        plt.scatter([], [], label=f"Tree Parameters:\n u: {u:.4f} \n d: {d:.4f} \n p: {p:.4f}")
        plt.legend(loc='lower left')

        plt.xticks(ticks=[i * dt for i in range(n + 1)], labels=[f'{i * dt:.1f}' for i in range(n + 1)])
        plt.xlim(-dt, n * dt + dt)

        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img.seek(0)

        return img
    
    def european_trinomial(self, n: int) -> tuple:
        """
        Price a European option using the trinomial tree model.
        """
        dt = self.T / n
        u = np.exp(self.sigma * np.sqrt(2 * dt))
        d = 1 / u

        pu = ((np.exp((self.r - self.q) * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) / 
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))) ** 2
        pd = ((np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp((self.r - self.q) * dt / 2)) / 
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))) ** 2
        pm = 1 - pu - pd

        stock_tree = np.zeros((2 * n + 1, n + 1))
        option_tree = np.zeros((2 * n + 1, n + 1))

        stock_tree[n][0] = self.S

        for i in range(1, n + 1):
            for j in range(-i, i + 1):
                stock_tree[n + j][i] = self.S * (u ** max(j, 0)) * (d ** max(-j, 0))
                print(f"Stock Price at node ({n + j}, {i}): {stock_tree[n + j][i]}")

        for j in range(-n, n + 1):
            if self.option_type.lower() == 'call':
                option_tree[n + j][n] = max(stock_tree[n + j][n] - self.K, 0)
            elif self.option_type.lower() == 'put':
                option_tree[n + j][n] = max(self.K - stock_tree[n + j][n], 0)
            print(f"Payoff for Option at node ({n + j}, {n}): {option_tree[n + j][n]}")

        for i in range(n - 1, -1, -1):
            for j in range(-i, i + 1):
                option_tree[n + j][i] = np.exp(-self.r * dt) * (
                    pu * option_tree[n + j + 1][i + 1] + 
                    pm * option_tree[n + j][i + 1] + 
                    pd * option_tree[n + j - 1][i + 1]
                )
                print(f"Option Price at node ({n + j}, {i}): {option_tree[n + j][i]}")

        return option_tree, stock_tree, u, d, pu, pd, pm, dt

    
    def draw_trinomial_tree(self, stock_tree, option_tree, n, u, d, pu, pd, pm, dt):
        G = nx.DiGraph()

        for i in range(n + 1):
            for j in range(-i, i + 1):
                stock_price = stock_tree[n - j][i]
                if i == n:
                    payoff = option_tree[n - j][n]
                    label = f"Asset: \n${stock_price:.2f}\nPayoff: \n${payoff:.2f}"
                    node_color = 'salmon'
                else:
                    option_price = option_tree[n - j][i]
                    label = f"Asset: \n${stock_price:.2f}\nOption: \n${option_price:.2f}"
                    node_color = 'lightgreen'

                G.add_node((i, j), label=label, color=node_color)

                if i < n:
                    G.add_edge((i, j), (i + 1, j + 1))  
                    G.add_edge((i, j), (i + 1, j))      
                    G.add_edge((i, j), (i + 1, j - 1))  

        pos = {(x, y): (x * dt, -y) for x, y in G.nodes()}

        plt.figure(figsize=(12, 8))
        colors = [G.nodes[node]['color'] for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=3500, node_color=colors, font_size=10,
                labels=nx.get_node_attributes(G, 'label'))

        plt.axis('off')
        
        plt.scatter([], [], label=f"Tree Parameters:\n u: {u:.4f} \n d: {d:.4f} \n pu: {pu:.4f} \n pd: {pd:.4f} \n pm: {pm:.4f}")
        plt.legend(loc='lower left')

        plt.xticks(ticks=[i * dt for i in range(n + 1)], labels=[f'{i * dt:.1f}' for i in range(n + 1)])
        plt.xlim(-dt, n * dt + dt)

        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img.seek(0)

        return img

    def monte_carlo_european(self, n_paths: int , n_steps: int) -> tuple:
        """
        Calculate the option price using Monte Carlo simulation and plot the generated paths.

        Args:
            n_paths (int): Number of Monte Carlo paths to simulate.
            n_steps (int): Number of time steps for each path.
            
        Returns:
            tuple: Option price and matplotlib figure of generated paths.
        """
        dt = self.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S
        
        for i in range(n_paths):
            for t in range(1, n_steps + 1):
                Z = np.random.normal(0,1)
                paths[i, t] = paths[i, t - 1] * np.exp((self.r - self.q - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)

        if self.option_type == self.CALL:
            payoff = np.maximum(paths[:, -1] - self.K, 0)
        elif self.option_type == self.PUT:
            payoff = np.maximum(self.K - paths[:, -1], 0)

        price = np.exp(-self.r * self.T) * np.mean(payoff)
        
        plt.figure(figsize=(12, 6))

        for j in range(n_paths):
            plt.plot(paths[j], alpha=0.5)
            
        strike_price_label = f'Strike Price: {self.K:.2f}'
        mean_price_label = f'Mean Final Price: {np.mean(paths[:, -1]):.2f}'

        plt.title('Generated Paths from European Monte Carlo Simulation')
        plt.xlabel('Time Steps')
        plt.ylabel('Asset Price')
        plt.axhline(y=self.K, color='red', linestyle='--', label=strike_price_label)
        plt.axhline(y=np.mean(paths[:, -1]), color='green', linestyle='--', label=mean_price_label)
        plt.legend(loc='upper left')
        plt.grid()

        return price, plt

    def _d1_d2(self):
        """Calculate d1 and d2 used in the Black-Scholes model."""
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
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


    def EU_models_comparison(self, n_steps_list=100, n_paths=1000) -> tuple:
        """
        Compare various European option pricing models.
        """
        bsm_prices = []
        merton_jump_diffusion_prices = []
        binomial_prices = []
        trinomial_prices = []
        monte_carlo_prices = []

        option = EUop(self.S, self.K, self.T, self.r, self.q, self.sigma, option_type=self.option_type)

        for n_steps in range(1, n_steps_list + 1):
            bsm_price = option.black_scholes()
            bsm_prices.append(bsm_price)

            merton_price = option.merton_jump_diffusion(mu_j=0.3, sigma_j=0.4, lam=1, max_iter=5000, stop_cond=1e-15)
            merton_jump_diffusion_prices.append(merton_price)

            binomial_price_tuple = option.european_binomial(n_steps)
            binomial_price = binomial_price_tuple[0][0,0]
            binomial_prices.append(binomial_price)

            trinomial_price_tuple = option.european_trinomial(n_steps)
            trinomial_price = trinomial_price_tuple[0][n_steps,0]
            trinomial_prices.append(trinomial_price)

            mc_price_tuple = option.monte_carlo_european(n_paths, n_steps)
            mc_price = mc_price_tuple[0]
            monte_carlo_prices.append(mc_price)

        return (bsm_prices, merton_jump_diffusion_prices, binomial_prices, trinomial_prices, monte_carlo_prices)