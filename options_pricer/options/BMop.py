import numpy as np
from scipy.stats import norm
from enum import Enum
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO

class OptionType(Enum):
    CALL = "call"
    PUT = "put"
    
class BMop:
    CALL = "call"
    PUT = "put"
    
    def __init__(self, S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType):
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
        self.option_type = option_type.lower()
        
    def BermudanBinomial(self, exercise_steps: int, n: int = 30) -> tuple:
        """
        Price a Bermudan option using the Cox-Ross-Rubinstein binomial model.
        """
        
        if exercise_steps > n:
            raise ValueError("Number of exercise steps cannot exceed the number of time steps.")
        
        exercise_steps_arr = np.linspace(0, n, exercise_steps, endpoint=False).astype(int)

        dt = self.T / n
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)

        if not 0 <= p <= 1:
            raise ValueError("Arbitrage detected: check inputs for no-arbitrage conditions.")

        asset_prices = np.zeros((n + 1, n + 1))
        continuation_prices = np.zeros((n + 1, n + 1))
        exercise_values = np.zeros((n + 1, n + 1))

        for i in range(n + 1):
            for j in range(i + 1):
                asset_prices[j, i] = self.S * (u ** (i - j)) * (d ** j)

        for j in range(n + 1):
            if self.option_type == 'call':
                continuation_prices[j, n] = max(0, asset_prices[j, n] - self.K)
            else:
                continuation_prices[j, n] = max(0, self.K - asset_prices[j, n])
                
            exercise_values[j, n] = max(0, asset_prices[j, n] - self.K) if self.option_type == 'call' \
                else max(0, self.K - asset_prices[j, n])

        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                continuation_value = np.exp(-self.r * dt) * (
                    p * continuation_prices[j, i + 1] + (1 - p) * continuation_prices[j + 1, i + 1]
                )
                exercise_value = max(0, asset_prices[j, i] - self.K) if self.option_type == 'call' \
                else max(0, self.K - asset_prices[j, i])
                
                if i in exercise_steps_arr:
                    continuation_prices[j, i] = max(exercise_value, continuation_value)
                else:
                    continuation_prices[j, i] = continuation_value
                
                exercise_values[j, i] = exercise_value

        return continuation_prices, asset_prices, exercise_values, u, d, p, dt

    def draw_binomial_tree(self, asset_prices, continuation_prices, exercise_values, n, u, d, p, dt):
        G = nx.DiGraph()
        
        for i in range(n+1):
            for j in range(i+1):
                stock_price = asset_prices[j,i]
                label = f'Asset:\n ${stock_price:.2f}'
                
                exercise_value = exercise_values[j, i]
                option_value = continuation_prices[j, i]
                
                if i == n:
                    label += f'\nPayoff:\n ${option_value:.2f}'
                    node_color = 'salmon'
                
                else:
                    if exercise_value == option_value and exercise_value > 0:
                        label += f"\nPayoff: \n ${exercise_value:.2f}"
                        node_color = 'salmon'
                    else:
                        if exercise_value == 0 and option_value == 0:
                            label += "\nOption: \n $0.00"
                        else:
                            label += f"\nOption:\n ${option_value:.2f}"
                        node_color = 'lightblue'

                G.add_node((i, j), label=label, color=node_color)
                
                if i < n:
                    G.add_edge((i, j), (i+1, j+1))
                    G.add_edge((i, j), (i+1, j))
                    
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
    
    def BermudanTrinomial(self, exercise_steps: int, n: int = 30):
        """
        Price a Bermudan option using the trinomial model.
        """
        
        if exercise_steps > n:
            raise ValueError("Number of exercise steps cannot exceed the number of time steps.")
        
        exercise_steps_arr = np.linspace(0, n, exercise_steps, endpoint=False).astype(int)

        dt = self.T / n
        u = np.exp(self.sigma * np.sqrt(2 * dt))
        d = 1 / u
        pu = ((np.exp((self.r - self.q) * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) / 
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2))))**2
        pd = ((np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp((self.r - self.q) * dt / 2)) / 
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2))))**2
        pm = 1 - pu - pd

        asset_prices = np.zeros((2 * n + 1, n + 1))
        continuation_prices = np.zeros((2 * n + 1, n + 1))
        exercise_values = np.zeros((2 * n + 1, n + 1))
        
        asset_prices[n, 0] = self.S

        for i in range(1, n + 1):
            for j in range(-i, i + 1):
                asset_prices[n + j, i] = self.S * (u ** max(j, 0)) * (d ** max(-j, 0))

        for j in range(-n, n + 1):
            if self.option_type == 'call':
                continuation_prices[n + j, n] = max(asset_prices[n + j, n] - self.K, 0)
            elif self.option_type == 'put':
                continuation_prices[n + j, n] = max(self.K - asset_prices[n + j, n], 0)

            exercise_values[n + j, n] = max(asset_prices[n + j, n] - self.K, 0) if self.option_type == 'call' \
                else max(self.K - asset_prices[n + j, n], 0)


        for i in range(n - 1, -1, -1):
            for j in range(-i, i + 1):
                continuation_value = np.exp(-self.r * dt) * (
                    pu * continuation_prices[n + j + 1, i + 1] + 
                    pm * continuation_prices[n + j, i + 1] + 
                    pd * continuation_prices[n + j - 1, i + 1]
                )
                exercise_value = max(asset_prices[n + j, i] - self.K, 0) if self.option_type == 'call' \
                    else max(self.K - asset_prices[n + j, i], 0)
                    
                if i in exercise_steps_arr:
                    continuation_prices[n +j, i] = max(exercise_value, continuation_value)
                else:
                    continuation_prices[n + j, i] = continuation_value
                    
                exercise_values[n + j, i] = exercise_value

        return continuation_prices, asset_prices, exercise_values, u, d, pu, pm, pd, dt
    
    def draw_trinomial_tree(self, asset_prices, continuation_prices, exercise_values, n, u, d, pu, pm, pd, dt):
        G = nx.DiGraph()

        for i in range(n + 1):
            for j in range(-i, i + 1):
                stock_price = asset_prices[n - j, i]
                label = f"Asset: \n ${stock_price:.2f}"

                exercise_value = exercise_values[n - j, i]
                option_value = continuation_prices[n - j, i]

                if i == n:
                    label += f"\nPayoff: \n ${option_value:.2f}"
                    node_color = 'salmon'
                else:
                    if exercise_value == option_value and exercise_value > 0:
                        label += f"\nPayoff: \n ${exercise_value:.2f}"
                        node_color = 'salmon'
                    else:
                        if exercise_value == 0 and option_value == 0:
                            label += "\nOption: \n $0.00"
                        else:
                            label += f"\nOption: \n ${option_value:.2f}"
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
        plt.scatter([], [], label=f"Tree Parameters:\n u: {u:.4f} \n d: {d:.4f} \n pu: {pu:.4f} \n pm: {pm:.4f} \n pd: {pd:.4f}")
        plt.legend(loc='lower left')
        plt.xticks(ticks=[i * dt for i in range(n + 1)], labels=[f'{i * dt:.1f}' for i in range(n + 1)])
        plt.xlim(-dt, n * dt + dt)

        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img.seek(0)

        return img
    
    def monte_carlo_bermudan(self, exercise_steps: int, n_paths: int, n_steps: int) -> tuple:
        """
        Calculate the Bermudan option price using Monte Carlo simulation with Least-Squares 
        regression (LSMC) and plot the generated paths.
        """
        
        if exercise_steps > n_steps:
            raise ValueError("Number of exercise steps cannot exceed the number of time steps.")
        
        dt = self.T / n_steps
        discount_factor = np.exp(-self.r * dt)
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S
        exercise_steps_arr = np.linspace(0, n_steps, exercise_steps, endpoint=False).astype(int)

        for i in range(n_paths):
            for t in range(1, n_steps + 1):
                Z = np.random.normal()
                paths[i, t] = paths[i, t - 1] * np.exp((self.r - self.q - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)

        if self.option_type == self.CALL:
            cash_flows = np.maximum(paths[:, -1] - self.K, 0)
        else:
            cash_flows = np.maximum(self.K - paths[:, -1], 0)

        for t in range(n_steps - 1, -1, -1):
            if t in exercise_steps_arr:
                if self.option_type == self.CALL:
                    exercise_value = np.maximum(paths[:, t] - self.K, 0)
                else:
                    exercise_value = np.maximum(self.K - paths[:, t], 0)

                in_the_money = exercise_value > 0
                
                if np.any(in_the_money):
                    X = paths[in_the_money, t]
                    Y = cash_flows[in_the_money] * discount_factor
                    
                    regression = np.polyfit(X, Y, 3)
                    continuation_value = np.polyval(regression, X)

                    exercise_now = exercise_value[in_the_money] > continuation_value
                    cash_flows[in_the_money] = np.where(exercise_now, exercise_value[in_the_money], 
                                                        cash_flows[in_the_money] * discount_factor)
            else:
                cash_flows *= discount_factor

        option_price = np.mean(cash_flows) * discount_factor

        plt.figure(figsize=(12, 6))
        for j in range(n_paths):
            plt.plot(paths[j], alpha=0.5)
        
        plt.title('Generated Paths from Bermudan Monte Carlo Simulation')
        plt.xlabel('Time Steps')
        plt.ylabel('Asset Price')
        plt.axhline(y=self.K, color='red', linestyle='--', label=f'Strike Price: {self.K:.2f}')
        plt.axhline(y=np.mean(paths[:, -1]), color='green', linestyle='--', label=f'Mean Final Price: {np.mean(paths[:, -1]):.2f}')
        plt.legend(loc='upper left')
        plt.grid()

        return option_price, plt


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


    def BM_models_comparison(self, exercise_steps=30, n_steps_list=100, n_paths=1000) -> tuple:
            """
            Compare various Bermudan option pricing models.
            """
            
            binomial_prices = []
            trinomial_prices = []
            monte_carlo_prices = []

            option = BMop(self.S, self.K, self.T, self.r, self.q, self.sigma, option_type=self.option_type)

            for n_steps in range(1, n_steps_list + 1):
                
                if exercise_steps > n_steps:
                    print(f"Skipping n_steps = {n_steps} because exercise_steps > n_steps.")
                    continue

                binomial_price_tuple = option.BermudanBinomial(exercise_steps, n_steps)
                binomial_price = binomial_price_tuple[0][0,0]
                binomial_prices.append(binomial_price)

                trinomial_price_tuple = option.BermudanTrinomial(exercise_steps, n_steps)
                trinomial_price = trinomial_price_tuple[0][n_steps,0]
                trinomial_prices.append(trinomial_price)

                mc_price_tuple = option.monte_carlo_bermudan(exercise_steps, n_paths, n_steps)
                mc_price = mc_price_tuple[0]
                monte_carlo_prices.append(mc_price)

            return (binomial_prices, trinomial_prices, monte_carlo_prices)