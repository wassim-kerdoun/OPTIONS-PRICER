import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_navigation_bar import st_navbar
import yfinance as yf
import pandas as pd
import datetime
from options import EUop,AMop,BMop,ImpliedVolatility
import plotly.graph_objects as go

def main():
    # -------------------------------
    # Streamlit Page Configuration
    # -------------------------------
    st.set_page_config(
        page_title="Option Pricing Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # -------------------------------
    # Sidebar Inputs
    # -------------------------------

    with st.sidebar:
        option_style = option_menu("Option Style", ["European", "American","Bermudan"],
                                icons=['calculator', 'calculator','calculator'], default_index=0)

        if option_style == "European":
            st.session_state['option_style'] = "EU"
        elif option_style == "American":
            st.session_state['option_style'] = "AM"
        elif option_style == "Bermudan":
            st.session_state['option_style'] = "BM"

    # -------------------------------
    # Main Page Content
    # -------------------------------

    if option_style == 'European':
        
        styles = {
            "nav": {
                "background-color": "rgb(235, 64, 52)",
            },
            "div": {
                "max-width": "32rem",
            },
            "span": {
                "border-radius": "0.5rem",
                "color": "rgb(49, 51, 63)",
                "margin": "0 0.125rem",
                "padding": "0.4375rem 0.625rem",
            },
            "active": {
                "background-color": "rgba(255, 255, 255, 0.25)",
            },
            "hover": {
                "background-color": "rgba(255, 255, 255, 0.35)",
            },
        }
        
        nav_bar = st_navbar(["Models Evaluation",'Models Comparison'],styles=styles)
        
        if nav_bar == "Models Evaluation":
            st.title("European Option Pricing Dashboard")

            pages = ["Black-Scholes-Merton", "Merton Jump Diffusion", "Binomial Trees", "Trinomial Trees", "Monte Carlo"]
            with st.expander("Select Method", expanded=True):
                method_type = st.selectbox("Choose a method", pages)

            if method_type == "Black-Scholes-Merton":

                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")

                if submit_button:
                    opc_call = EUop(
                        S=current_price,
                        K=strike_price,
                        T=time_to_maturity,
                        r=risk_free_rate,
                        q=dividend_yield,
                        sigma=volatility,
                        option_type=EUop.CALL
                    )
                    opc_put = EUop(
                        S=current_price,
                        K=strike_price,
                        T=time_to_maturity,
                        r=risk_free_rate,
                        q=dividend_yield,
                        sigma=volatility,
                        option_type=EUop.PUT
                    )

                    call_price = opc_call.black_scholes()
                    put_price = opc_put.black_scholes()

                    delta_call = opc_call.delta()
                    delta_put = opc_put.delta()
                    
                    st.markdown("---")
                    
                    st.header("Computation Results")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Call Option Price")
                        st.write(f"${call_price:.2f}")

                    with col2:
                        st.subheader("Put Option Price")
                        st.write(f"${put_price:.2f}")

                    st.markdown("---")
                    st.header("Option Greeks")
                    greeks = {
                        "Delta Call": delta_call,
                        "Delta Put": delta_put,
                        "Theta Call": opc_call.theta(),
                        "Theta Put": opc_put.theta(),
                        "Rho Call": opc_call.rho(),
                        "Rho Put": opc_put.rho(),
                        "Gamma": opc_call.gamma(),
                        "Vega": opc_call.vega(),
                    }
                    
                    greeks_df = pd.DataFrame(greeks, index=['Values'])
                    
                    def color_values(val):
                        if val > 0:
                            color = 'color: green;'
                        elif val < 0:
                            color = 'color: red;'
                        elif val == 0:
                            color = 'color: black;'
                        return color


                    styled_greeks_df = (
                        greeks_df.style
                        .format("{:.4f}")
                        .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                        .applymap(color_values)
                    )

                    st.table(styled_greeks_df)
                    
                    st.markdown('---')
                    
                    st.header("Implied Volatility Computations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if impl_vol_method == 'Least Squares':
                            call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                    opc_call.q, opc_call.sigma, 
                                                    opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                    
                            st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                            
                        elif impl_vol_method == 'Newton-Raphson':
                            call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                            max_iterations,tolerance)
                                    
                            st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                    
                    with col2:    
                        if impl_vol_method == 'Least Squares':
                            put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                opc_put.q, opc_put.sigma, 
                                                opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                    
                            st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                            
                        elif impl_vol_method == 'Newton-Raphson':
                            put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                opc_put.q, opc_put.sigma, 
                                                opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                            max_iterations,tolerance)
                                    
                            st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                            
                            
                    st.markdown('---')
                    call_option_price_plot = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).option_price_plot()
                    put_option_price_plot = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                opc_put.q, opc_put.sigma, 
                                                opc_put.option_type).option_price_plot()
                    
                    st.header("Option Price Plots")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Call Option Price")
                        st.plotly_chart(call_option_price_plot)
                    
                    with col2:
                        st.subheader("Put Option Price")
                        st.plotly_chart(put_option_price_plot)
                    
                    st.markdown('---')
                    
                    st.header("Option Price Surface")
                    k_values, t_values, price_matrix = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).price_surface()
                    
                    call_option_price_surface = go.Figure(data=[go.Surface(x=k_values, y=t_values, z=price_matrix,
                                                    colorscale='Viridis', showscale=True)])

                    call_option_price_surface.update_layout(
                        scene=dict(
                            xaxis_title='Strike Price',
                            yaxis_title='Time to Expiration (Years)',
                            zaxis_title='Price'
                        )
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Call Option Price Surface")
                        st.plotly_chart(call_option_price_surface)
                        
                    k_values, t_values, price_matrix = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                opc_put.q, opc_put.sigma, 
                                                opc_put.option_type).price_surface()
                    
                    put_option_price_surface = go.Figure(data=[go.Surface(x=k_values, y=t_values, z=price_matrix,
                                                    colorscale='Viridis', showscale=True)])
                    
                    put_option_price_surface.update_layout(
                        scene=dict(
                            xaxis_title='Strike Price',
                            yaxis_title='Time to Expiration (Years)',
                            zaxis_title='Price'
                        )
                    )
                    with col2:
                        st.subheader("Put Option Price Surface")
                        st.plotly_chart(put_option_price_surface)
                    
                    st.markdown('---')
                    
                    st.subheader("Volatility Smile")
                    
                    Ks, implied_vols = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).volatility_smile()
                    
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=Ks / current_price, y=implied_vols, mode='lines+markers', name='Implied Volatility'))
                    fig1.update_layout(
                        xaxis_title="Moneyness (K/S)",
                        yaxis_title="Implied Volatility",
                        xaxis=dict(showgrid=True),
                        yaxis=dict(showgrid=True)
                        )
                    st.plotly_chart(fig1)
                    
                    st.markdown('---')
                    
                    st.subheader('Volatility Term Structure')
                    
                    T_values, implied_vols = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).volatility_term_structure()
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=T_values, y=implied_vols, mode='lines+markers', name='Implied Volatility'))
                    fig2.update_layout(
                        xaxis_title="Time to Expiration (Years)",
                        yaxis_title="Implied Volatility",
                        xaxis=dict(showgrid=True),
                        yaxis=dict(showgrid=True)
                    )
                    st.plotly_chart(fig2)
                    
                    st.markdown('---')
                    
                    col1, col2 = st.columns(2)
                    
                    K_values, T_values, implied_vol_matrix = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).volatility_surface()
                    
                    fig3 = go.Figure(data=[go.Surface(x=K_values / current_price, y=T_values, z=implied_vol_matrix,
                                                    colorscale='Viridis', showscale=True)])

                    fig3.update_layout(
                        scene=dict(
                            xaxis_title='Moneyness (K/S)',
                            yaxis_title='Time to Expiration (Years)',
                            zaxis_title='Implied Volatility'
                        )
                    )
                    
                    with col1:
                        st.subheader('Volatility Surface')
                        st.plotly_chart(fig3)
                    
                    st.markdown('---')
                    
                    ks, ts, delta_matrix = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).delta_surface()
                    
                    fig5 = go.Figure(data=[go.Surface(x=ks / current_price, y=ts, z=delta_matrix,
                                                    colorscale='Viridis', showscale=True)])

                    fig5.update_layout(
                        scene=dict(
                            xaxis_title='Moneyness (K/S)',
                            yaxis_title='Time to Expiration (Years)',
                            zaxis_title='Delta'
                        )
                    )

                    with col2:
                        st.subheader('Delta Surface')
                        st.plotly_chart(fig5)
                        
                    kk, tt, gamma_matrix = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).gamma_surface()
                    
                    fig6 = go.Figure(data=[go.Surface(x=kk / current_price, y=tt, z=gamma_matrix,
                                                    colorscale='Viridis', showscale=True)])

                    fig6.update_layout(
                        scene=dict(
                            xaxis_title='Moneyness (K/S)',
                            yaxis_title='Time to Expiration (Years)',
                            zaxis_title='Gamma'
                        )
                    )

                    with col1:
                        st.subheader('Gamma Surface')
                        st.plotly_chart(fig6)
                        
                    kk, tt, vega_matrix = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).vega_surface()
                    
                    fig7 = go.Figure(data=[go.Surface(x=kk / current_price, y=tt, z=vega_matrix,
                                                    colorscale='Viridis', showscale=True)])

                    fig7.update_layout(
                        scene=dict(
                            xaxis_title='Moneyness (K/S)',
                            yaxis_title='Time to Expiration (Years)',
                            zaxis_title='Vega'
                        )
                    )

                    with col2:
                        st.subheader('Vega Surface')
                        st.plotly_chart(fig7)
                        
                    kk, tt, theta_matrix = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).theta_surface()
                    
                    fig8 = go.Figure(data=[go.Surface(x=kk / current_price, y=tt, z=theta_matrix,
                                                    colorscale='Viridis', showscale=True)])

                    fig8.update_layout(
                        scene=dict(
                            xaxis_title='Moneyness (K/S)',
                            yaxis_title='Time to Expiration (Years)',
                            zaxis_title='Theta'
                        )
                    )

                    with col1:
                        st.subheader('Theta Surface')
                        st.plotly_chart(fig8)
    
           
            elif method_type == 'Merton Jump Diffusion':
                
                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                    #mu_j,sigma_j,lam,max_iter
                with col2:
                    with st.expander("Parameters for Merton Jump Diffusion",expanded=True):
                        mu_j = st.number_input("Jump Mean (mu_j)",value=0.0,min_value=0.0,step=0.01)
                        sigma_j = st.number_input("Jump Volatility (sigma_j)",value=0.15,min_value=0.00,step=0.01)
                        lam = st.number_input("Jump Rate (lam)",value=0.50,min_value=0.00,step=0.01)
                        max_iter = st.number_input("Max Iterations",value=100,min_value=0,max_value=1000,step=1)
                        
                        st.write("Jump Mean: The expected value of the jump size in the logarithm distribution of the stock price.")
                        st.write("Jump Volatility: The standard deviation of the jump size in the logarithm distribution of the stock price.")
                        st.write("Jump Rate: The expected value of the jump rate of the stock price. \n lam=0.1 (1 jump per 10 years)")
                    
                with col1:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                submit_button = st.button("Submit")
                
                if submit_button:
                    opc_call = EUop(
                        S=current_price,
                        K=strike_price,
                        T=time_to_maturity,
                        r=risk_free_rate,
                        q=dividend_yield,
                        sigma=volatility,
                        option_type=EUop.CALL
                    )
                    opc_put = EUop(
                        S=current_price,
                        K=strike_price,
                        T=time_to_maturity,
                        r=risk_free_rate,
                        q=dividend_yield,
                        sigma=volatility,
                        option_type=EUop.PUT
                    )
                    
                    call_price = opc_call.merton_jump_diffusion(mu_j,sigma_j,lam,max_iter)
                    put_price = opc_put.merton_jump_diffusion(mu_j,sigma_j,lam,max_iter)
                    
                    st.header("Computation Results")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Call Option Price")
                        st.write(f"${call_price:.2f}")

                    with col2:
                        st.subheader("Put Option Price")
                        st.write(f"${put_price:.2f}")

                    st.markdown("---")
                    st.header("Option Greeks")
                    greeks = {
                        "Delta Call": opc_call.delta(),
                        "Delta Put": opc_put.delta(),
                        "Theta Call": opc_call.theta(),
                        "Theta Put": opc_put.theta(),
                        "Rho Call": opc_call.rho(),
                        "Rho Put": opc_put.rho(),
                        "Gamma": opc_call.gamma(),
                        "Vega": opc_call.vega(),
                    }
                    
                    greeks_df = pd.DataFrame(greeks, index=['Values'])
                    
                    def color_values(val):
                        if val > 0:
                            color = 'color: green;'
                        elif val < 0:
                            color = 'color: red;'
                        elif val == 0:
                            color = 'color: black;'
                        return color


                    styled_greeks_df = (
                        greeks_df.style
                        .format("{:.4f}")
                        .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                        .applymap(color_values)
                    )

                    st.table(styled_greeks_df)
                    
                    st.markdown('---')
                    
                    st.header("Implied Volatility Computations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if impl_vol_method == 'Least Squares':
                            call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                    opc_call.q, opc_call.sigma, 
                                                    opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                    
                            st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                            
                        elif impl_vol_method == 'Newton-Raphson':
                            call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                opc_call.q, opc_call.sigma, 
                                                opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                            max_iterations,tolerance)
                                    
                            st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                    
                    with col2:    
                        if impl_vol_method == 'Least Squares':
                            put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                opc_put.q, opc_put.sigma, 
                                                opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                    
                            st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                            
                        elif impl_vol_method == 'Newton-Raphson':
                            put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                opc_put.q, opc_put.sigma, 
                                                opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                            max_iterations,tolerance)
                                    
                            st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                            
                            
                    # st.markdown('---')
                    
                    # st.subheader("Volatility Smile")
                    
                    # Ks, implied_vols = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                    #                             opc_call.q, opc_call.sigma, 
                    #                             opc_call.option_type).volatility_smile()
                    
                    # fig1 = go.Figure()
                    # fig1.add_trace(go.Scatter(x=Ks / current_price, y=implied_vols, mode='lines+markers', name='Implied Volatility'))
                    # fig1.update_layout(
                    #     xaxis_title="Moneyness (K/S)",
                    #     yaxis_title="Implied Volatility",
                    #     xaxis=dict(showgrid=True),
                    #     yaxis=dict(showgrid=True)
                    #     )
                    # st.plotly_chart(fig1)
                    
                    # st.markdown('---')
                    
                    # st.subheader('Volatility Term Structure')
                    
                    # T_values, implied_vols = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                    #                             opc_call.q, opc_call.sigma, 
                    #                             opc_call.option_type).volatility_term_structure()
                    
                    # fig2 = go.Figure()
                    # fig2.add_trace(go.Scatter(x=T_values, y=implied_vols, mode='lines+markers', name='Implied Volatility'))
                    # fig2.update_layout(
                    #     xaxis_title="Time to Expiration (Years)",
                    #     yaxis_title="Implied Volatility",
                    #     xaxis=dict(showgrid=True),
                    #     yaxis=dict(showgrid=True)
                    # )
                    # st.plotly_chart(fig2)
                    
                    # st.markdown('---')
                    
                    # st.subheader('Volatility Surface')
                    
                    # K_values, T_values, implied_vol_matrix = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                    #                             opc_call.q, opc_call.sigma, 
                    #                             opc_call.option_type).volatility_surface()
                    
                    # fig3 = go.Figure(data=[go.Surface(x=K_values / current_price, y=T_values, z=implied_vol_matrix,
                    #                                 colorscale='Viridis', showscale=True)])

                    # fig3.update_layout(
                    #     scene=dict(
                    #         xaxis_title='Moneyness (K/S)',
                    #         yaxis_title='Time to Expiration (Years)',
                    #         zaxis_title='Implied Volatility'
                    #     )
                    # )

                    # st.plotly_chart(fig3)
                    
                    # st.subheader('Price Surface')
                    
                    # k_values, t_values, price_matrix = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r,
                    #                             opc_call.q, opc_call.sigma, 
                    #                             opc_call.option_type).price_surface()
                    
                    # fig4 = go.Figure(data=[go.Surface(x=k_values / current_price, y=t_values, z=price_matrix,
                    #                                 colorscale='Viridis', showscale=True)])

                    # fig4.update_layout(
                    #     scene=dict(
                    #         xaxis_title='Moneyness (K/S)',
                    #         yaxis_title='Time to Expiration (Years)',
                    #         zaxis_title='Price'
                    #     )
                    # )

                    # st.plotly_chart(fig4)
                
                
            elif method_type == 'Binomial Trees':
                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                
                with col1:
                    tree_step = st.number_input("Tree Step (n)", value=30, min_value=1, step=1)
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")
                
                st.markdown('---')
                
                try:
                    if submit_button:
                        opc_call = EUop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.CALL
                        )
                        opc_put = EUop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.PUT
                        )
                        
                        call_price,call_binomial_matrix,u,d,p,dt = opc_call.european_binomial(tree_step)
                        put_price,put_binomial_matrix,_,_,_,_ = opc_put.european_binomial(tree_step)
                        
                        st.header("Computation Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Call Option Price")
                            st.write(f"${call_price[0][0]:.2f}")

                        with col2:
                            st.subheader("Put Option Price")
                            st.write(f"${put_price[0][0]:.2f}")

                        st.markdown("---")
                        st.header("Option Greeks")
                        greeks = {
                            "Delta Call": opc_call.delta(),
                            "Delta Put": opc_put.delta(),
                            "Theta Call": opc_call.theta(),
                            "Theta Put": opc_put.theta(),
                            "Rho Call": opc_call.rho(),
                            "Rho Put": opc_put.rho(),
                            "Gamma": opc_call.gamma(),
                            "Vega": opc_call.vega(),
                        }
                        
                        greeks_df = pd.DataFrame(greeks, index=['Values'])
                        
                        def color_values(val):
                            if val > 0:
                                color = 'color: green;'
                            elif val < 0:
                                color = 'color: red;'
                            elif val == 0:
                                color = 'color: black;'
                            return color


                        styled_greeks_df = (
                            greeks_df.style
                            .format("{:.4f}")
                            .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                            .applymap(color_values)
                        )

                        st.table(styled_greeks_df)
                        
                        st.markdown('---')
                    
                        st.header("Implied Volatility Computations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if impl_vol_method == 'Least Squares':
                                call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                        opc_call.q, opc_call.sigma, 
                                                        opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                        
                                st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                    opc_call.q, opc_call.sigma, 
                                                    opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                        
                        with col2:    
                            if impl_vol_method == 'Least Squares':
                                put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                        
                                st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                        
                        st.markdown('---')
                        st.header(f'{opc_call.option_type.capitalize()} Option Binomial Tree')
                        
                        tree1 = opc_call.draw_binomial_tree(call_binomial_matrix, call_price, tree_step, u, d, p, dt)
                        
                        st.image(tree1,use_column_width=True)
                        
                        st.header(f'{opc_put.option_type.capitalize()} Option Binomial Tree')
                        
                        tree2 = opc_put.draw_binomial_tree(put_binomial_matrix, put_price, tree_step, u, d, p, dt)
                        
                        st.image(tree2,use_column_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
            elif method_type == 'Trinomial Trees':
                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                
                with col1:
                    tree_step = st.number_input("Tree Step (n)", value=30, min_value=1, step=1)
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")
                
                st.markdown('---')
                
                try:
                    if submit_button:
                        opc_call = EUop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.CALL
                        )
                        opc_put = EUop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.PUT
                        )
                        
                        call_price,call_binomial_matrix,u,d,pu,p_d,pm,dt = opc_call.european_trinomial(tree_step)
                        put_price,put_binomial_matrix,_,_,_,_,_,_ = opc_put.european_trinomial(tree_step)
                        
                        st.header("Computation Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Call Option Price")
                            st.write(f"${call_price[tree_step][0]:.2f}")

                        with col2:
                            st.subheader("Put Option Price")
                            st.write(f"${put_price[tree_step][0]:.2f}")

                        st.markdown("---")
                        st.header("Option Greeks")
                        greeks = {
                            "Delta Call": opc_call.delta(),
                            "Delta Put": opc_put.delta(),
                            "Theta Call": opc_call.theta(),
                            "Theta Put": opc_put.theta(),
                            "Rho Call": opc_call.rho(),
                            "Rho Put": opc_put.rho(),
                            "Gamma": opc_call.gamma(),
                            "Vega": opc_call.vega(),
                        }
                        
                        greeks_df = pd.DataFrame(greeks, index=['Values'])
                        
                        def color_values(val):
                            if val > 0:
                                color = 'color: green;'
                            elif val < 0:
                                color = 'color: red;'
                            elif val == 0:
                                color = 'color: black;'
                            return color


                        styled_greeks_df = (
                            greeks_df.style
                            .format("{:.4f}")
                            .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                            .applymap(color_values)
                        )

                        st.table(styled_greeks_df)
                        
                        st.markdown('---')
                    
                        st.header("Implied Volatility Computations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if impl_vol_method == 'Least Squares':
                                call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                        opc_call.q, opc_call.sigma, 
                                                        opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                        
                                st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                    opc_call.q, opc_call.sigma, 
                                                    opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                        
                        with col2:    
                            if impl_vol_method == 'Least Squares':
                                put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                        
                                st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                        
                        st.markdown('---')
                        st.header(f'{opc_call.option_type.capitalize()} Option Triromial Tree')
                        
                        tree3 = opc_call.draw_trinomial_tree(call_binomial_matrix, call_price, tree_step,
                                                            u, d, pu, p_d, pm, dt)
                        
                        st.image(tree3,use_column_width=True)
                        
                        st.header(f'{opc_put.option_type.capitalize()} Option Triromial Tree')
                        
                        tree4 = opc_put.draw_trinomial_tree(put_binomial_matrix, put_price, tree_step, 
                                                            u, d, pu, p_d, pm, dt)
                        
                        st.image(tree4,use_column_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
            elif method_type == 'Monte Carlo':
                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                
                with col1:
                    n_paths = st.number_input("Number of paths to generate (n)", value=10000, min_value=1, step=1)
                    
                with col1:
                    n_steps = st.number_input("Number of time steps (m)", value=100, min_value=1, step=1)
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")
                
                st.markdown('---')
                
                try:
                    if submit_button:
                        opc_call = EUop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.CALL
                        )
                        opc_put = EUop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.PUT
                        )
                        
                        call_price, call_paths = opc_call.monte_carlo_european(n_paths, n_steps)
                        put_price, _ = opc_put.monte_carlo_european(n_paths, n_steps)
                        
                        st.header("Computation Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Call Option Price")
                            st.write(f"${call_price:.2f}")

                        with col2:
                            st.subheader("Put Option Price")
                            st.write(f"${put_price:.2f}")

                        st.markdown("---")
                        st.header("Option Greeks")
                        greeks = {
                            "Delta Call": opc_call.delta(),
                            "Delta Put": opc_put.delta(),
                            "Theta Call": opc_call.theta(),
                            "Theta Put": opc_put.theta(),
                            "Rho Call": opc_call.rho(),
                            "Rho Put": opc_put.rho(),
                            "Gamma": opc_call.gamma(),
                            "Vega": opc_call.vega(),
                        }
                        
                        greeks_df = pd.DataFrame(greeks, index=['Values'])
                        
                        def color_values(val):
                            if val > 0:
                                color = 'color: green;'
                            elif val < 0:
                                color = 'color: red;'
                            elif val == 0:
                                color = 'color: black;'
                            return color


                        styled_greeks_df = (
                            greeks_df.style
                            .format("{:.4f}")
                            .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                            .applymap(color_values)
                        )

                        st.table(styled_greeks_df)
                        
                        st.markdown('---')
                    
                        st.header("Implied Volatility Computations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if impl_vol_method == 'Least Squares':
                                call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                        opc_call.q, opc_call.sigma, 
                                                        opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                        
                                st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                    opc_call.q, opc_call.sigma, 
                                                    opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                        
                        with col2:    
                            if impl_vol_method == 'Least Squares':
                                put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                        
                                st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                                
                        st.markdown('---')
                        
                        st.header('Generated Paths')
                        
                        st.pyplot(call_paths)
                        
                except Exception as e:
                    st.error(e)
                    
        elif nav_bar == 'Models Comparison':
            st.title("Models Comparison Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

            with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

            with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


            with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

            with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
            with col2:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
            
            with col2:
                    n_steps = st.number_input("Number of time steps (m)", value=100, min_value=1, step=1) 
                       
            with col2:
                    n_paths = st.number_input("Number of paths to generate for Monte Carlo Simulation(n)",
                                              value=1000, min_value=1, step=1)
                    
                    
            submit_button = st.button("Submit")
                
            st.markdown('---')
                
            try:
                if submit_button:
                        opc_call = EUop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.CALL
                        )
                        opc_put = EUop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.PUT
                        )
                        
                
                        call_bsm_prices, call_merton_jump_diffusion_prices, call_binomial_prices, call_trinomial_prices, call_monte_carlo_prices = opc_call.EU_models_comparison(n_steps, n_paths)

                        put_bsm_prices, put_merton_jump_diffusion_prices, put_binomial_prices, put_trinomial_prices, put_monte_carlo_prices = opc_put.EU_models_comparison(n_steps, n_paths)

                        def plot_convergence(bsm_prices, merton_prices, binomial_prices, trinomial_prices, monte_carlo_prices,call=True):
                            step_range = list(range(1, len(bsm_prices) + 1))

                            fig = go.Figure()

                            fig.add_trace(go.Scatter(x=step_range, y=bsm_prices, mode='lines+markers', name='Black-Scholes',opacity=0.5))
                            fig.add_trace(go.Scatter(x=step_range, y=merton_prices, mode='lines', name='Merton Jump Diffusion',opacity=0.5))
                            fig.add_trace(go.Scatter(x=step_range, y=binomial_prices, mode='lines+markers', name='Binomial Tree'))
                            fig.add_trace(go.Scatter(x=step_range, y=trinomial_prices, mode='lines+markers', name='Trinomial Tree'))
                            fig.add_trace(go.Scatter(x=step_range, y=monte_carlo_prices, mode='lines+markers', name='Monte Carlo'))
                            
                            if call:
                                fig.update_layout(
                                    title='Price Convergence of European Call Option Pricing Models',
                                    xaxis_title='Number of Steps',
                                    yaxis_title='Option Price',
                                    legend_title='Pricing Methods',
                                    template='plotly_white'
                                )

                                return fig
                            
                            else:
                                fig.update_layout(
                                    title='Price Convergence of European Put Option Pricing Models',
                                    xaxis_title='Number of Steps',
                                    yaxis_title='Option Price',
                                    legend_title='Pricing Methods',
                                    template='plotly_white'
                                )

                                return fig
                        
                        st.plotly_chart(plot_convergence(call_bsm_prices, call_merton_jump_diffusion_prices, 
                                                        call_binomial_prices, call_trinomial_prices, 
                                                        call_monte_carlo_prices, call=True))
                        
                        st.plotly_chart(plot_convergence(put_bsm_prices, put_merton_jump_diffusion_prices,
                                                        put_binomial_prices, put_trinomial_prices,
                                                        put_monte_carlo_prices, call=False))
            
            except ValueError as e:
                st.error(f"Error: {e}")
            
            
    elif option_style == 'American':
        
        styles = {
            "nav": {
                "background-color": "rgb(235, 64, 52)",
            },
            "div": {
                "max-width": "32rem",
            },
            "span": {
                "border-radius": "0.5rem",
                "color": "rgb(49, 51, 63)",
                "margin": "0 0.125rem",
                "padding": "0.4375rem 0.625rem",
            },
            "active": {
                "background-color": "rgba(255, 255, 255, 0.25)",
            },
            "hover": {
                "background-color": "rgba(255, 255, 255, 0.35)",
            },
        }
        
        nav_bar = st_navbar(["Models Evaluation", "Models Comparison"],styles=styles)
        
        if nav_bar == 'Models Evaluation':
        
            st.title("American Option Pricing Dashboard")

            pages = ["Binomial Trees", "Trinomial Trees", "Monte Carlo"]
            
            with st.expander("Select Method", expanded=True):
                method_type = st.selectbox("Choose a method", pages)
            
            if method_type == 'Binomial Trees':
                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                
                with col1:
                    tree_step = st.number_input("Tree Step (n)", value=30, min_value=1, step=1)
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")
                
                st.markdown('---')
                
                try:
                    if submit_button:
                        opc_call = AMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.CALL,
                        )
                        opc_put = AMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.PUT
                        )
                        
                        call_continuation_price,call_binomial_matrix,call_exercise_values,u,d,p,dt = opc_call.AmericanBinomial(tree_step)
                        put_continuation_price,put_binomial_matrix,put_exercise_values,_,_,_,_ = opc_put.AmericanBinomial(tree_step)
                        
                        st.header("Computation Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Call Option Price")
                            st.write(f"${call_continuation_price[0,0]:.2f}")

                        with col2:
                            st.subheader("Put Option Price")
                            st.write(f"${put_continuation_price[0,0]:.2f}")

                        st.markdown("---")
                        st.header("Option Greeks")
                        greeks = {
                            "Delta Call": opc_call.delta(),
                            "Delta Put": opc_put.delta(),
                            "Theta Call": opc_call.theta(),
                            "Theta Put": opc_put.theta(),
                            "Rho Call": opc_call.rho(),
                            "Rho Put": opc_put.rho(),
                            "Gamma": opc_call.gamma(),
                            "Vega": opc_call.vega(),
                        }
                        
                        greeks_df = pd.DataFrame(greeks, index=['Values'])
                        
                        def color_values(val):
                            if val > 0:
                                color = 'color: green;'
                            elif val < 0:
                                color = 'color: red;'
                            elif val == 0:
                                color = 'color: black;'
                            return color


                        styled_greeks_df = (
                            greeks_df.style
                            .format("{:.4f}")
                            .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                            .applymap(color_values)
                        )

                        st.table(styled_greeks_df)
                        
                        st.markdown('---')
                    
                        st.header("Implied Volatility Computations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if impl_vol_method == 'Least Squares':
                                call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                        opc_call.q, opc_call.sigma, 
                                                        opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                        
                                st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                    opc_call.q, opc_call.sigma, 
                                                    opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                        
                        with col2:    
                            if impl_vol_method == 'Least Squares':
                                put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                        
                                st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                        
                        st.markdown('---')
                        st.header(f'{opc_call.option_type.capitalize()} Option Binomial Tree')
                        
                        tree5 = opc_call.draw_binomial_tree(call_binomial_matrix, call_continuation_price,
                                                            call_exercise_values,tree_step, u, d, p, dt)
                        
                        st.image(tree5,use_column_width=True)
                        
                        st.header(f'{opc_put.option_type.capitalize()} Option Binomial Tree')
                        
                        tree6 = opc_put.draw_binomial_tree(put_binomial_matrix, put_continuation_price,
                                                        put_exercise_values, tree_step, u, d, p, dt)
                        
                        st.image(tree6,use_column_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
            elif method_type == 'Trinomial Trees':
                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                
                with col1:
                    tree_step = st.number_input("Tree Step (n)", value=30, min_value=1, step=1)
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")
                
                st.markdown('---')
                
                try:
                    if submit_button:
                        opc_call = AMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.CALL,
                        )
                        opc_put = AMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=EUop.PUT
                        )
                        
                        call_continuation_price,call_binomial_matrix,call_exercise_values,u,d,pu,pm,p_d,dt = opc_call.AmericanTrinomial(tree_step)
                        put_continuation_price,put_binomial_matrix,put_exercise_values,_,_,_,_,_,_ = opc_put.AmericanTrinomial(tree_step)
                        
                        st.header("Computation Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Call Option Price")
                            st.write(f"${call_continuation_price[tree_step,0]:.2f}")

                        with col2:
                            st.subheader("Put Option Price")
                            st.write(f"${put_continuation_price[tree_step,0]:.2f}")

                        st.markdown("---")
                        st.header("Option Greeks")
                        greeks = {
                            "Delta Call": opc_call.delta(),
                            "Delta Put": opc_put.delta(),
                            "Theta Call": opc_call.theta(),
                            "Theta Put": opc_put.theta(),
                            "Rho Call": opc_call.rho(),
                            "Rho Put": opc_put.rho(),
                            "Gamma": opc_call.gamma(),
                            "Vega": opc_call.vega(),
                        }
                        
                        greeks_df = pd.DataFrame(greeks, index=['Values'])
                        
                        def color_values(val):
                            if val > 0:
                                color = 'color: green;'
                            elif val < 0:
                                color = 'color: red;'
                            elif val == 0:
                                color = 'color: black;'
                            return color


                        styled_greeks_df = (
                            greeks_df.style
                            .format("{:.4f}")
                            .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                            .applymap(color_values)
                        )

                        st.table(styled_greeks_df)
                        
                        st.markdown('---')
                    
                        st.header("Implied Volatility Computations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if impl_vol_method == 'Least Squares':
                                call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                        opc_call.q, opc_call.sigma, 
                                                        opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                        
                                st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                    opc_call.q, opc_call.sigma, 
                                                    opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                        
                        with col2:    
                            if impl_vol_method == 'Least Squares':
                                put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                        
                                st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                        
                        st.markdown('---')
                        st.header(f'{opc_call.option_type.capitalize()} Option Trinomial Tree')
                        
                        tree7 = opc_call.draw_trinomial_tree(call_binomial_matrix, call_continuation_price,
                                                            call_exercise_values,tree_step, u, d, pu, pm, p_d, dt)
                        
                        st.image(tree7,use_column_width=True)
                        
                        st.header(f'{opc_put.option_type.capitalize()} Option Trinomial Tree')
                        
                        tree8 = opc_put.draw_trinomial_tree(put_binomial_matrix, put_continuation_price,
                                                            put_exercise_values,tree_step, u, d, pu, pm, p_d, dt)
                        
                        st.image(tree8,use_column_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
            elif method_type == 'Monte Carlo':
                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                
                with col1:
                    n_paths = st.number_input("Number of paths to generate (n)", value=10000, min_value=1, step=1)
                    
                with col1:
                    n_steps = st.number_input("Number of time steps (m)", value=100, min_value=1, step=1)
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")
                
                st.markdown('---')
                
                try:
                    if submit_button:
                        opc_call = AMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=AMop.CALL
                        )
                        opc_put = AMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=AMop.PUT
                        )
                        
                        call_price, call_paths = opc_call.monte_carlo_american(n_paths, n_steps)
                        put_price, _ = opc_put.monte_carlo_american(n_paths, n_steps)
                        
                        st.header("Computation Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Call Option Price")
                            st.write(f"${call_price:.2f}")

                        with col2:
                            st.subheader("Put Option Price")
                            st.write(f"${put_price:.2f}")

                        st.markdown("---")
                        st.header("Option Greeks")
                        greeks = {
                            "Delta Call": opc_call.delta(),
                            "Delta Put": opc_put.delta(),
                            "Theta Call": opc_call.theta(),
                            "Theta Put": opc_put.theta(),
                            "Rho Call": opc_call.rho(),
                            "Rho Put": opc_put.rho(),
                            "Gamma": opc_call.gamma(),
                            "Vega": opc_call.vega(),
                        }
                        
                        greeks_df = pd.DataFrame(greeks, index=['Values'])
                        
                        def color_values(val):
                            if val > 0:
                                color = 'color: green;'
                            elif val < 0:
                                color = 'color: red;'
                            elif val == 0:
                                color = 'color: black;'
                            return color


                        styled_greeks_df = (
                            greeks_df.style
                            .format("{:.4f}")
                            .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                            .applymap(color_values)
                        )

                        st.table(styled_greeks_df)
                        
                        st.markdown('---')
                    
                        st.header("Implied Volatility Computations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if impl_vol_method == 'Least Squares':
                                call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                        opc_call.q, opc_call.sigma, 
                                                        opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                        
                                st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                    opc_call.q, opc_call.sigma, 
                                                    opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                        
                        with col2:    
                            if impl_vol_method == 'Least Squares':
                                put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                        
                                st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                                
                        st.markdown('---')
                        
                        st.header('Generated Paths')
                        
                        st.pyplot(call_paths)
                        
                except Exception as e:
                    st.error(e)
         
        elif nav_bar == "Models Comparison":
            st.title("American Options Pricing Models Comparison Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

            with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

            with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


            with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

            with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
            with col2:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
            
            with col2:
                    n_steps = st.number_input("Number of time steps (m)", value=100, min_value=1, step=1) 
                       
            with col2:
                    n_paths = st.number_input("Number of paths to generate for Monte Carlo Simulation(n)",
                                              value=1000, min_value=1, step=1)
                    
                    
            submit_button = st.button("Submit")
                
            st.markdown('---')
                
            try:
                if submit_button:
                        opc_call = AMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=AMop.CALL
                        )
                        opc_put = AMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=AMop.PUT
                        )
                        
                
                        call_binomial_prices, call_trinomial_prices, call_monte_carlo_prices = opc_call.AM_models_comparison(n_steps, n_paths)

                        put_binomial_prices, put_trinomial_prices, put_monte_carlo_prices = opc_put.AM_models_comparison(n_steps, n_paths)

                        def plot_convergence(binomial_prices, trinomial_prices, monte_carlo_prices,call=True):
                            step_range = list(range(1, len(binomial_prices) + 1))

                            fig = go.Figure()

                            fig.add_trace(go.Scatter(x=step_range, y=binomial_prices, mode='lines+markers', name='Binomial Tree'))
                            fig.add_trace(go.Scatter(x=step_range, y=trinomial_prices, mode='lines+markers', name='Trinomial Tree'))
                            fig.add_trace(go.Scatter(x=step_range, y=monte_carlo_prices, mode='lines+markers', name='Monte Carlo'))
                            
                            if call:
                                fig.update_layout(
                                    title='Price Convergence of American Call Option Pricing Models',
                                    xaxis_title='Number of Steps',
                                    yaxis_title='Option Price',
                                    legend_title='Pricing Methods',
                                    template='plotly_white'
                                )

                                return fig
                            
                            else:
                                fig.update_layout(
                                    title='Price Convergence of American Put Option Pricing Models',
                                    xaxis_title='Number of Steps',
                                    yaxis_title='Option Price',
                                    legend_title='Pricing Methods',
                                    template='plotly_white'
                                )

                                return fig
                        
                        st.plotly_chart(plot_convergence(call_binomial_prices, call_trinomial_prices, 
                                                        call_monte_carlo_prices, call=True))
                        
                        st.plotly_chart(plot_convergence(put_binomial_prices, put_trinomial_prices,
                                                        put_monte_carlo_prices, call=False))
            
            except ValueError as e:
                st.error(f"Error: {e}")
                
    
    elif option_style == 'Bermudan':
        
        styles = {
            "nav": {
                "background-color": "rgb(235, 64, 52)",
            },
            "div": {
                "max-width": "32rem",
            },
            "span": {
                "border-radius": "0.7rem",
                "color": "rgb(49, 51, 63)",
                "margin": "0 0.125rem",
                "padding": "0.4375rem 0.625rem",
            },
            "active": {
                "background-color": "rgba(255, 255, 255, 0.25)",
            },
            "hover": {
                "background-color": "rgba(255, 255, 255, 0.35)",
            },
        }
        
        nav_bar = st_navbar(["Models Evaluation",'Models Comparison'],styles=styles)
        
        if nav_bar == 'Models Evaluation':
            st.title("Bermudan Option Pricing Dashboard")

            pages = ["Binomial Trees", "Trinomial Trees", "Monte Carlo"]
            with st.expander("Select Method", expanded=True):
                method_type = st.selectbox("Choose a method", pages)

            if method_type == "Binomial Trees":

                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                    
                with col1:
                    tree_step = st.number_input("Tree Steps", value=30, min_value=1, step=1)
                    
                with col1:
                    exercise_steps = st.number_input("Exercise Steps", value=30, min_value=1, step=1)
                    st.write('This number indicates how many times the option can be exercised during its lifetime.')
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")
                
                try:
                    if submit_button:
                        opc_call = BMop(S=current_price,
                                        K=strike_price,
                                        T=time_to_maturity,
                                        r=risk_free_rate,
                                        q=dividend_yield,
                                        sigma=volatility,
                                        option_type=BMop.CALL)
                        
                        opc_put = BMop(S=current_price,
                                        K=strike_price,
                                        T=time_to_maturity,
                                        r=risk_free_rate,
                                        q=dividend_yield,
                                        sigma=volatility,
                                        option_type=BMop.PUT)
                        
                        call_continuation_price, call_binomial_matrix, call_exercise_values, u, d, p, dt = opc_call.BermudanBinomial(exercise_steps,tree_step)
                        put_continuation_price, put_binomial_matrix, put_exercise_values, u, d, p, dt = opc_put.BermudanBinomial(exercise_steps,tree_step)
                        
                        
                        st.header("Computation Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Call Option Price")
                            st.write(f"${call_continuation_price[0,0]:.2f}")

                        with col2:
                            st.subheader("Put Option Price")
                            st.write(f"${put_continuation_price[0,0]:.2f}")

                        st.markdown("---")
                        st.header("Option Greeks")
                        greeks = {
                                "Delta Call": opc_call.delta(),
                                "Delta Put": opc_put.delta(),
                                "Theta Call": opc_call.theta(),
                                "Theta Put": opc_put.theta(),
                                "Rho Call": opc_call.rho(),
                                "Rho Put": opc_put.rho(),
                                "Gamma": opc_call.gamma(),
                                "Vega": opc_call.vega(),
                            }
                            
                        greeks_df = pd.DataFrame(greeks, index=['Values'])
                            
                        def color_values(val):
                                if val > 0:
                                    color = 'color: green;'
                                elif val < 0:
                                    color = 'color: red;'
                                elif val == 0:
                                    color = 'color: black;'
                                return color


                        styled_greeks_df = (
                                greeks_df.style
                                .format("{:.4f}")
                                .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                                .applymap(color_values)
                            )

                        st.table(styled_greeks_df)
                            
                        st.markdown('---')
                        
                        st.header("Implied Volatility Computations")
                            
                        col1, col2 = st.columns(2)
                            
                        with col1:
                                if impl_vol_method == 'Least Squares':
                                    call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                            opc_call.q, opc_call.sigma, 
                                                            opc_call.option_type).implied_volatility_least_squares(call_market_price)
                                            
                                    st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                                    
                                elif impl_vol_method == 'Newton-Raphson':
                                    call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                        opc_call.q, opc_call.sigma, 
                                                        opc_call.option_type).implied_volatility_newton_raphson(call_market_price,initial_guess,
                                                                                                                    max_iterations,tolerance)
                                            
                                    st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                            
                        with col2:    
                                if impl_vol_method == 'Least Squares':
                                    put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                        opc_put.q, opc_put.sigma, 
                                                        opc_put.option_type).implied_volatility_least_squares(put_market_price)
                                            
                                    st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                                    
                                elif impl_vol_method == 'Newton-Raphson':
                                    put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                        opc_put.q, opc_put.sigma, 
                                                        opc_put.option_type).implied_volatility_newton_raphson(put_market_price,initial_guess,
                                                                                                                    max_iterations,tolerance)
                                            
                                    st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                                    
                        st.markdown('---')
                            
                        st.header(f'{opc_call.option_type.capitalize()} Option Binomial Tree')
                        
                        tree9 = opc_call.draw_binomial_tree(call_binomial_matrix, call_continuation_price,
                                                            call_exercise_values,tree_step, u, d, p, dt)
                        
                        st.image(tree9,use_column_width=True)
                        
                        st.header(f'{opc_put.option_type.capitalize()} Option Binomial Tree')
                        
                        tree10 = opc_put.draw_binomial_tree(put_binomial_matrix, put_continuation_price,
                                                            put_exercise_values,tree_step, u, d, p, dt)
                        
                        st.image(tree10,use_column_width=True)
                            
                except Exception as e:
                    st.error(e)
            
            elif method_type == 'Trinomial Trees':
                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                    
                with col1:
                    tree_step = st.number_input("Tree Steps", value=30, min_value=1, step=1)
                    
                with col1:
                    exercise_steps = st.number_input("Exercise Steps", value=30, min_value=1, step=1)
                    st.write('This number indicates how many times the option can be exercised during its lifetime.')
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")
                
                try:
                    if submit_button:
                        opc_call = BMop(S=current_price,
                                        K=strike_price,
                                        T=time_to_maturity,
                                        r=risk_free_rate,
                                        q=dividend_yield,
                                        sigma=volatility,
                                        option_type=BMop.CALL)
                        
                        opc_put = BMop(S=current_price,
                                        K=strike_price,
                                        T=time_to_maturity,
                                        r=risk_free_rate,
                                        q=dividend_yield,
                                        sigma=volatility,
                                        option_type=BMop.PUT)
                        
                        call_continuation_price, call_binomial_matrix, call_exercise_values, u, d, pu, pm, p_d, dt = opc_call.BermudanTrinomial(exercise_steps,tree_step)
                        put_continuation_price, put_binomial_matrix, put_exercise_values, u, d, pu, pm, p_d, dt = opc_put.BermudanTrinomial(exercise_steps,tree_step)
                        
                        
                        st.header("Computation Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Call Option Price")
                            st.write(f"${call_continuation_price[tree_step,0]:.2f}")

                        with col2:
                            st.subheader("Put Option Price")
                            st.write(f"${put_continuation_price[tree_step,0]:.2f}")

                        st.markdown("---")
                        st.header("Option Greeks")
                        greeks = {
                                "Delta Call": opc_call.delta(),
                                "Delta Put": opc_put.delta(),
                                "Theta Call": opc_call.theta(),
                                "Theta Put": opc_put.theta(),
                                "Rho Call": opc_call.rho(),
                                "Rho Put": opc_put.rho(),
                                "Gamma": opc_call.gamma(),
                                "Vega": opc_call.vega(),
                            }
                            
                        greeks_df = pd.DataFrame(greeks, index=['Values'])
                            
                        def color_values(val):
                                if val > 0:
                                    color = 'color: green;'
                                elif val < 0:
                                    color = 'color: red;'
                                elif val == 0:
                                    color = 'color: black;'
                                return color


                        styled_greeks_df = (
                                greeks_df.style
                                .format("{:.4f}")
                                .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                                .applymap(color_values)
                            )

                        st.table(styled_greeks_df)
                            
                        st.markdown('---')
                        
                        st.header("Implied Volatility Computations")
                            
                        col1, col2 = st.columns(2)
                            
                        with col1:
                                if impl_vol_method == 'Least Squares':
                                    call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                            opc_call.q, opc_call.sigma, 
                                                            opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                            
                                    st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                                    
                                elif impl_vol_method == 'Newton-Raphson':
                                    call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                        opc_call.q, opc_call.sigma, 
                                                        opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                                    max_iterations,tolerance)
                                            
                                    st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                            
                        with col2:    
                                if impl_vol_method == 'Least Squares':
                                    put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                        opc_put.q, opc_put.sigma, 
                                                        opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                            
                                    st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                                    
                                elif impl_vol_method == 'Newton-Raphson':
                                    put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                        opc_put.q, opc_put.sigma, 
                                                        opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                                    max_iterations,tolerance)
                                            
                                    st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                                    
                        st.markdown('---')
                            
                        st.header(f'{opc_call.option_type.capitalize()} Option Trinomial Tree')
                        
                        tree11 = opc_call.draw_trinomial_tree(call_binomial_matrix, call_continuation_price,
                                                            call_exercise_values,tree_step, u, d, pu, pm, p_d, dt)
                        
                        st.image(tree11,use_column_width=True)
                        
                        st.header(f'{opc_put.option_type.capitalize()} Option Trinomial Tree')
                        
                        tree12 = opc_put.draw_trinomial_tree(put_binomial_matrix, put_continuation_price,
                                                            put_exercise_values,tree_step, u, d, pu, pm, p_d, dt)
                        
                        st.image(tree12,use_column_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                            
            elif method_type == 'Monte Carlo':
                col1, col2 = st.columns(2)

                with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

                with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

                with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


                with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

                with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
                with col1:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
                
                with col1:
                    n_paths = st.number_input("Number of paths to generate (n)", value=10000, min_value=1, step=1)
                    
                with col1:
                    n_steps = st.number_input("Number of time steps (m)", value=100, min_value=1, step=1)
                    
                with col1:
                    exercise_steps = st.number_input("Exercise Steps", value=30, min_value=1, step=1)
                    st.write('This number indicates how many times the option can be exercised during its lifetime.')
                    
                with col2:
                        with st.expander("Parameters for Implied volatility",expanded=True):
                            impl_vol_method = st.selectbox('Choose Method', ['Least Squares','Newton-Raphson'])
                            
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                # call_market_price = st.number_input('Call Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                call_market_price = options_chain.calls[options_chain.calls['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Call Market Price: ${call_market_price[0]:.2f}")
                                
                            with sub_col2:
                                # put_market_price = st.number_input('Put Market Price',min_value=0.0,max_value=999.0,step=0.01)
                                data = yf.Ticker(ticker)
                                options_date = data.options
                                current_date = datetime.datetime.now()
                                target_date = current_date + datetime.timedelta(time_to_maturity * 365)
                                expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in options_date]
                                closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                                options_chain = data.option_chain(closest_date.strftime('%Y-%m-%d'))
                                put_market_price = options_chain.puts[options_chain.puts['strike'] == strike_price]['lastPrice'].values
                                st.write(f"Put Market Price: ${put_market_price[0]:.2f}")
                                
                            if impl_vol_method == 'Newton-Raphson':
                                initial_guess = st.number_input('Initial Guess',min_value=0.0,max_value=1.0,value=0.25,step=0.01)
                                max_iterations = st.number_input('Max Iterations',min_value=0,max_value=1000,value=500,step=1)
                                tolerance = st.number_input('Tolerance',min_value=0.000001,max_value=1.00000,
                                                            value=0.000001,step=0.000001,format="%.6f")
                                
                                
                submit_button = st.button("Submit")
                
                st.markdown('---')
                
                try:
                    if submit_button:
                        opc_call = BMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=BMop.CALL
                        )
                        opc_put = BMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=BMop.PUT
                        )
                        
                        call_price, call_paths = opc_call.monte_carlo_bermudan(exercise_steps, n_paths, n_steps)
                        put_price, _ = opc_put.monte_carlo_bermudan(exercise_steps, n_paths, n_steps)
                        
                        st.header("Computation Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Call Option Price")
                            st.write(f"${call_price:.2f}")

                        with col2:
                            st.subheader("Put Option Price")
                            st.write(f"${put_price:.2f}")

                        st.markdown("---")
                        st.header("Option Greeks")
                        greeks = {
                            "Delta Call": opc_call.delta(),
                            "Delta Put": opc_put.delta(),
                            "Theta Call": opc_call.theta(),
                            "Theta Put": opc_put.theta(),
                            "Rho Call": opc_call.rho(),
                            "Rho Put": opc_put.rho(),
                            "Gamma": opc_call.gamma(),
                            "Vega": opc_call.vega(),
                        }
                        
                        greeks_df = pd.DataFrame(greeks, index=['Values'])
                        
                        def color_values(val):
                            if val > 0:
                                color = 'color: green;'
                            elif val < 0:
                                color = 'color: red;'
                            elif val == 0:
                                color = 'color: black;'
                            return color


                        styled_greeks_df = (
                            greeks_df.style
                            .format("{:.4f}")
                            .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
                            .applymap(color_values)
                        )

                        st.table(styled_greeks_df)
                        
                        st.markdown('---')
                    
                        st.header("Implied Volatility Computations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if impl_vol_method == 'Least Squares':
                                call_least_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                        opc_call.q, opc_call.sigma, 
                                                        opc_call.option_type).implied_volatility_least_squares(call_market_price[0])
                                        
                                st.write(f"Implied Volatility for Call Option: {call_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                call_newton_iv = ImpliedVolatility(opc_call.S, opc_call.K, opc_call.T, opc_call.r, 
                                                    opc_call.q, opc_call.sigma, 
                                                    opc_call.option_type).implied_volatility_newton_raphson(call_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Call Option: {call_newton_iv:.4f}")
                        
                        with col2:    
                            if impl_vol_method == 'Least Squares':
                                put_least_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r, 
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_least_squares(put_market_price[0])
                                        
                                st.write(f"Implied Volatility for Put Option: {put_least_iv:.4f}")
                                
                            elif impl_vol_method == 'Newton-Raphson':
                                put_newton_iv = ImpliedVolatility(opc_put.S, opc_put.K, opc_put.T, opc_put.r,
                                                    opc_put.q, opc_put.sigma, 
                                                    opc_put.option_type).implied_volatility_newton_raphson(put_market_price[0],initial_guess,
                                                                                                                max_iterations,tolerance)
                                        
                                st.write(f"Implied Volatility for Put Option: {put_newton_iv:.4f}")
                                
                        st.markdown('---')
                        
                        st.header('Generated Paths')
                        
                        st.pyplot(call_paths)
                        
                        
                except Exception as e:
                    st.error(e)      
         
        elif nav_bar == "Models Comparison":
            st.title("Bermudan Options Pricing Models Comparison Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
                    data = yf.Ticker(ticker)
                    hist = data.history(period="1d")
                    current_price = hist.iloc[0]['Close']
                    st.write(f'Current Price of {ticker}: ${current_price:.2f}')

            with col1:
                    strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)

            with col2:
                    with st.expander("Select Risk-Free Interest Rate", expanded=True):
                        duration_option = st.selectbox("Select Duration", ["3 months", "2 years", "5 years", "10 years", "Custom"])
                        
                        if duration_option == "Custom":
                            risk_free_rate = st.number_input("Enter Custom Risk-Free Rate (in %)", value=5.0, min_value=0.0, step=0.01)
                            risk_free_rate /= 100
                            st.write(f"Custom Risk-Free Interest Rate: {risk_free_rate:.2%}")
                        else:
                            duration_map = {
                                "3 months": "^IRX",  # 3-Month Treasury Bill
                                "2 years": "2YY=F",  # 2-Year Yield Futures
                                "5 years": "^FVX",   # 5-Year Treasury Yield
                                "10 years": "^TNX"    # 10-Year Treasury Note
                            }

                            tickerr = duration_map[duration_option]

                            try:
                                if duration_option == "2 years":
                                    data = yf.download(tickerr, period="5d", interval="1d")
                                else:
                                    data = yf.download(tickerr, period="1d", interval="1d")
                                if data.empty:
                                    raise ValueError("No data returned for the selected duration.")
                                risk_free_rate = data['Close'].iloc[-1] / 100
                                st.write(f"Risk-Free Interest Rate: {risk_free_rate:.2%}")
                            except Exception as e:
                                st.error(f"Error fetching risk-free rate for {duration_option}: {e}")
                                risk_free_rate = 0.05  # Default value if there's an error


            with col1:
                    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, step=0.01)

            with col1:
                    time_to_maturity = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.01, step=0.01)
                    
            with col2:
                    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
            
            with col2:
                    n_steps = st.number_input("Number of time steps (m)", value=100, min_value=1, step=1) 
                       
            with col2:
                    n_paths = st.number_input("Number of paths to generate for Monte Carlo Simulation(n)",
                                              value=1000, min_value=1, step=1)
                    
            with col1:
                    exercise_steps = st.number_input("Exercise Steps", value=30, min_value=1, step=1)
                    
                    
            submit_button = st.button("Submit")
                
            st.markdown('---')
                
            try:
                if submit_button:
                        opc_call = BMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=BMop.CALL
                        )
                        opc_put = BMop(
                            S=current_price,
                            K=strike_price,
                            T=time_to_maturity,
                            r=risk_free_rate,
                            q=dividend_yield,
                            sigma=volatility,
                            option_type=BMop.PUT
                        )
                        
                
                        call_binomial_prices, call_trinomial_prices, call_monte_carlo_prices = opc_call.BM_models_comparison(exercise_steps, 
                                                                                                                        n_steps, n_paths)

                        put_binomial_prices, put_trinomial_prices, put_monte_carlo_prices = opc_put.BM_models_comparison(exercise_steps,
                                                                                                                    n_steps, n_paths)

                        def plot_convergence(binomial_prices, trinomial_prices, monte_carlo_prices,call=True):
                            step_range = list(range(1, len(binomial_prices) + 1))

                            fig = go.Figure()

                            fig.add_trace(go.Scatter(x=step_range, y=binomial_prices, mode='lines+markers', name='Binomial Tree'))
                            fig.add_trace(go.Scatter(x=step_range, y=trinomial_prices, mode='lines+markers', name='Trinomial Tree'))
                            fig.add_trace(go.Scatter(x=step_range, y=monte_carlo_prices, mode='lines+markers', name='Monte Carlo'))
                            
                            if call:
                                fig.update_layout(
                                    title='Price Convergence of Bermudan Call Option Pricing Models',
                                    xaxis_title='Number of Steps',
                                    yaxis_title='Option Price',
                                    legend_title='Pricing Methods',
                                    template='plotly_white'
                                )

                                return fig
                            
                            else:
                                fig.update_layout(
                                    title='Price Convergence of Bermudan Put Option Pricing Models',
                                    xaxis_title='Number of Steps',
                                    yaxis_title='Option Price',
                                    legend_title='Pricing Methods',
                                    template='plotly_white'
                                )

                                return fig
                        
                        st.plotly_chart(plot_convergence(call_binomial_prices, call_trinomial_prices, 
                                                        call_monte_carlo_prices, call=True))
                        
                        st.plotly_chart(plot_convergence(put_binomial_prices, put_trinomial_prices,
                                                        put_monte_carlo_prices, call=False))
            
            except ValueError as e:
                st.error(f"Error: {e}")
                
                
                
if __name__ == '__main__':
    main()
