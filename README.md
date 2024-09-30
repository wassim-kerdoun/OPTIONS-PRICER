# Options Pricer

## Overview
**The Options Pricer** is a web application built with Python, designed for pricing various types of options using different pricing models. It includes implementations for European, American, and Bermudan options, allowing users to understand and analyze option pricing mechanisms effectively. The application also supports calculating the Greeks, which are essential for assessing risk and sensitivity in options trading.

## Features
- **Option Pricing Models**:
  - **Black-Scholes Model (BSM)**: A widely used model for pricing European options, providing closed-form solutions based on constant volatility and interest rates.
  - **Merton Jumps**: An extension of the Black-Scholes model that incorporates jumps in the price process, suitable for modeling stock prices with sudden movements, making it more applicable to real market conditions.
  - **Binomial Model (CRR)**: The Cox-Ross-Rubinstein model for pricing American and Bermudan options, allowing for flexible time steps and the ability to exercise options at different points.
  - **Trinomial Model**: An extension of the binomial model that provides more accurate pricing for American options by allowing for three possible price movements (up, down, and unchanged) at each time step.
  - **Monte Carlo Simulations**: Utilizes Monte Carlo methods to simulate paths for underlying assets and estimate option prices, particularly useful for complex derivatives where analytical solutions are difficult to obtain.

- **Volatility Analysis**:
  - **Volatility Smile**: Analyzes and visualizes the phenomenon where implied volatility varies with the strike price, often observed in equity options markets.
  - **Volatility Term Structure**: Examines how implied volatility changes over different maturities, providing insights into market expectations and risk.
  - **Volatility Surface**: Generates a 3D representation of implied volatility across various strike prices and maturities, offering a comprehensive view of market sentiment.

- **Greeks Calculation**: Provides functionality to calculate the Greeks (Delta, Gamma, Theta, Vega, and Rho) for options, helping traders assess risk and sensitivity to various market factors.

- **Implied Volatility**: Implements methods to estimate the implied volatility using various numerical methods, including Least Squares and Newton-Raphson, enhancing the accuracy of option pricing.

## Directory Structure
```
options_pricer/
│
├── __pycache__/
│
├── options/
│   ├── __init__.py
│   ├── AMop.py          # Contains classes and methods for American options pricing
│   ├── BMop.py          # Contains classes and methods for Bermudan options pricing
│   ├── EUop.py          # Contains classes and methods for European options pricing
│   ├── Greeks.py        # Functionality for calculating the Greeks of options
│   ├── ImpliedVolatility.py  # Methods for calculating implied volatility
│
├── main.py              # Main entry point for the web application
├── test.ipynb           # Jupyter notebook for testing functionalities
└── Directory_Structure.markdown  # Documentation on directory structure
```

## Future Enhancements
- **Complex Pricing Models**:
  - **Finite Difference Methods**: Implement numerical techniques to solve partial differential equations (PDEs) for option pricing, providing more flexibility and accuracy, especially for options with complex features.
  - **Heston Model**: Incorporate the Heston model for stochastic volatility, allowing for more realistic modeling of asset prices that exhibit volatility clustering and mean reversion.
  - **Local Volatility Models**: Add local volatility models to capture the dynamics of implied volatility surfaces and improve pricing accuracy for exotic options.
  
- **User Interface**: A more user-friendly web interface for easier navigation and input.
- **Data Visualization**: Enhanced graphical representations of pricing models, Greek sensitivities, and volatility analyses.
