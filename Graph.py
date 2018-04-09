import Finance as ff
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def draw_portfolios_omega(ddf,ticker,optimal=None):
    df=ddf
    returns_annual = df.mean() * 251
    port_returns = []
    port_volatility = []
    omega_ratio = []
    stock_weights = []
    num_assets = df.shape[1]
    num_portfolios = 500
    np.random.seed(102)   #101
    omega2=0
    # populate the empty lists with each portfolios returns,risk and weights
    for portfolio in range(num_portfolios):
        omega=0
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = ff.portfolio_vol(df,weights.T)
        for i in range (0,df.shape[1]):
            omega = weights.T[i]* ff.omega2(df.iloc[:,i].as_matrix())+omega

        omega_ratio.append(omega)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)


    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Omega Ratio': omega_ratio}

    for counter, symbol in enumerate(ticker):
        portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]
    df = pd.DataFrame(portfolio)
    column_order = ['Returns', 'Volatility', 'Omega Ratio'] + [stock + ' Weight' for stock in ticker]
    df = df[column_order]
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Omega Ratio',  cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
    v=plt.axis()
    if optimal is not None:

        for i in range(0, ddf.shape[1]):
            omega2 = optimal[i] * ff.omega2(ddf.iloc[:, i].as_matrix()) + omega2
        ret = np.dot(optimal, returns_annual)
        volatility = ff.portfolio_vol(ddf, optimal)
        plt.scatter(x=volatility, y=ret, c='blue', marker='x')
    plt.xlabel('Odchylenie standardowe')
    plt.ylabel('Oczekiwana stopa zwrotu')
    plt.title(build_description(ticker,optimal))
    plt.annotate(str(omega2),xy=(volatility,ret))
    ax=plt.subplot()
    str2 ='Target:'+str(ff.gtarget*100)+'%'
    ax.text(v[0],v[3]*0.8, str2, style='italic',fontsize=16
            , bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 0.0}
           )

    plt.show()


def build_description(ticker,optimal):
    string=''
    i=0
    optimal2=[]
    for ti in ticker:


        optimal2.append("{0:.3f}%".format(optimal[i] * 100))
        if i==5:
            string=string+' \n '
        string=string+ti+':'+optimal2[i]+' '
        i=i+1

    return string




def draw_portfolios_sharpe(ddf,ticker,optimal=None):
    df = ddf
    returns_annual = df.mean() * 251
    port_returns = []
    port_volatility = []
    omega_ratio = []
    stock_weights = []
    num_assets = df.shape[1]
    num_portfolios = 500
    np.random.seed(101)
    omega2 = 0
    # populate the empty lists with each portfolios returns,risk and weights
    for portfolio in range(num_portfolios):
        omega = 0
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = ff.portfolio_vol(df, weights.T)
        for i in range(0, df.shape[1]):
            omega = weights.T[i] * ff.sharpe_ratio(df.iloc[:, i].as_matrix()) + omega
        omega_ratio.append(omega)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Sharpe Ratio': omega_ratio}

    for counter, symbol in enumerate(ticker):
        portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]
    df = pd.DataFrame(portfolio)
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in ticker]
    df = df[column_order]
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio', cmap='gnuplot', edgecolors='black', figsize=(10, 8),
                    grid=True)
    if optimal is not None:
        for i in range(0, ddf.shape[1]):
            omega2 = optimal[i] * ff.sharpe_ratio(ddf.iloc[:, i].as_matrix()) + omega2
        ret = np.dot(optimal, returns_annual)
        volatility = ff.portfolio_vol(ddf, optimal)
        plt.scatter(x=volatility, y=ret, c='yellow', marker='x')
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Random Portfolios')
    plt.show()


def draw_portfolios_calmar(ddf,ticker,optimal=None):
    df=ddf
    returns_annual = df.mean() * 251
    port_returns = []
    port_volatility = []
    omega_ratio = []
    stock_weights = []
    num_assets = df.shape[1]
    num_portfolios = 500
    np.random.seed(101)
    omega2 = 0
    # populate the empty lists with each portfolios returns,risk and weights
    for portfolio in range(num_portfolios):
        omega = 0
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = ff.portfolio_vol(df, weights.T)
        for i in range(0, df.shape[1]):
            omega = weights.T[i] * ff.calmar_ratio(df.iloc[:, i].as_matrix()) + omega
        omega_ratio.append(omega)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)


    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Calmar Ratio': omega_ratio}

    for counter, symbol in enumerate(ticker):
        portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]
    df = pd.DataFrame(portfolio)
    #  fig, ax = plt.subplots()
    column_order = ['Returns', 'Volatility', 'Calmar Ratio'] + [stock + ' Weight' for stock in ticker]
    df = df[column_order]
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Calmar Ratio', cmap='RdYlGn', edgecolors='black', figsize=(10, 8),
                    grid=True)
    if optimal is not None:
        for i in range(0, ddf.shape[1]):
            omega2 = optimal[i] * ff.calmar_ratio(ddf.iloc[:, i].as_matrix()) + omega2

        ret = np.dot(optimal, returns_annual)
        volatility = ff.portfolio_vol(ddf, optimal)
        plt.scatter(x=volatility, y=ret, c='blue', marker='x')
    plt.scatter(x=volatility, y=ret, c='blue', marker='x')
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    # plt.clabel('Omega Ratio')
    plt.title('Efficient Frontier')
    plt.show()
