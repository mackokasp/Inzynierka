import Finance as ff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='kasprzyk_maciej', api_key='Sv8dKdTSuABfjDOiQ1l5')

def draw_portfolios_omega(ddf,ticker,optimal=None):
    df=ddf
    returns_annual = df.mean()
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

        omega = ff.portfolio_omega(df,weights.T)

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



        omega2 =  ff.portfolio_omega(ddf,optimal)
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


def draw_table(df,optimal):
    print (optimal)

    res = prepare_portfolio_data(df)
    opt_res = prepare_optimal(df,optimal)

    trace = go.Table(
        header=dict(values=['   ','Market(SP500)', 'Optimal'],
                    line=dict(color='#7D7F80'),
                    fill=dict(color='#a1c3d1'),
                    align=['left'] * 10),
        cells=dict(values=[['Volatility','Omega' ,'Sharpe' ,'Sortino' ,'Calamr','Max Drowdown','Treynor','Evar','Information_Ratio','Upside_Potential'],
                           res,opt_res],
                   line=dict(color='#7D7F80'),
                   fill=dict(color='#EDFAFF'),
                   align=['left'] * 10))

    layout = dict(width=1500, height=3000)
    data = [trace]
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='styled_table')




def prepare_optimal(df,opt):
    port_returns = []
    port_volatility = []
    omega_ratio = []
    sortino_ratio = []
    calmar_ratio = []
    sharpe_ratio = []
    maxdd_ratio = []
    treynor_ratio = []
    information_ratio=[]
    upside_ratio=[]
    evar_ratio = []
    result = []
    omega = 0
    sortino = 0
    calmar = 0
    sharpe = 0
    maxdd = 0
    treynor = 0
    evar = 0
    inf=0
    upside=0
    volatility = ff.portfolio_vol(df, opt)
    m = np.random.uniform(-0.03, 0.05, df.shape[0])
    for i in range(0, df.shape[1]):
        paper = df.iloc[:, i].as_matrix()
        e = np.mean(paper)
        omega = opt[i] * ff.omega2(paper) + omega
        sortino = opt[i] * ff.sortino_ratio(paper) + sortino
        calmar = opt[i] * ff.calmar_ratio(paper) + calmar
        sharpe = opt[i] * ff.sharpe_ratio(paper) + sharpe
        evar = opt[i] * ff.excess_var(e, paper) + evar
        maxdd = opt[i] * ff.max_dd(paper) + maxdd
        treynor = opt[i] * ff.treynor_ratio(e, paper, m) + treynor
        inf= opt[i] * ff.information_ratio( paper, m) + inf
        upside=opt[i]*ff.upside_potential_ratio(paper) + upside




        omega_ratio.append(omega)
        sharpe_ratio.append(sharpe)
        calmar_ratio.append(calmar)
        sortino_ratio.append(sortino)
        maxdd_ratio.append(maxdd)
        treynor_ratio.append(treynor)
        evar_ratio.append(evar)
        port_volatility.append(volatility)
        information_ratio.append(inf)
        upside_ratio.append(upside)

    result.append(np.mean(port_volatility))
    result.append(omega)
    result.append(np.mean(sharpe_ratio))
    result.append(np.mean(sortino_ratio))
    result.append(np.mean(calmar_ratio))
    result.append(np.mean(maxdd_ratio))
    result.append(np.mean(treynor_ratio))
    result.append(np.mean(evar_ratio))
    result.append(np.mean(information_ratio))
    result.append(np.mean(upside_ratio))
    return result





def prepare_portfolio_data (df):
    returns_annual = df.mean() * 251
    port_returns = []
    port_volatility = []
    omega_ratio = []
    sortino_ratio=[]
    calmar_ratio=[]
    sharpe_ratio=[]
    maxdd_ratio=[]
    treynor_ratio=[]
    evar_ratio=[]
    result= []
    information_ratio=[]
    upside_ratio=[]




    stock_weights = []
    num_assets = df.shape[1]
    print(df.shape[0])
    num_portfolios = 15
    np.random.seed(102)
    m = np.random.uniform(-0.03, 0.05, df.shape[0])

    # populate the empty lists with each portfolios returns,risk and weights
    for portfolio in range(num_portfolios):
        omega = 0
        sortino=0
        calmar=0
        sharpe=0
        maxdd =0
        treynor=0
        evar =0
        inf=0
        upside = 0

        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = ff.portfolio_vol(df, weights.T)
        omega=ff.portfolio_omega(df,weights)
        for i in range(0, df.shape[1]):
            paper = df.iloc[:, i].as_matrix()
            e=np.mean(paper)
            #omega = weights.T[i] * ff.omega2(paper) + omega
            sortino=weights.T[i] * ff.sortino_ratio(paper) + sortino
            calmar = weights.T[i] * ff.calmar_ratio(paper) + calmar
            sharpe = weights.T[i] * ff.sharpe_ratio(paper) + sharpe
            evar= weights.T[i] * ff.excess_var(e,paper) + evar
            maxdd=weights.T[i] * ff.max_dd(paper) + maxdd
            treynor =weights.T[i] * ff.treynor_ratio(e,paper,m) +treynor
            inf = weights.T[i] * ff.information_ratio(paper, m) + inf
            upside = weights.T[i] * ff.upside_potential_ratio(paper) + upside



        omega_ratio.append(omega)
        sharpe_ratio.append(sharpe)
        calmar_ratio.append(calmar)
        sortino_ratio.append(sortino)
        maxdd_ratio.append(maxdd)
        treynor_ratio.append(treynor)
        evar_ratio.append(evar)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
        information_ratio.append(inf)
        upside_ratio.append(upside)

    result.append(np.mean(port_volatility))
    result.append(np.mean(omega_ratio))
    result.append(np.mean(sharpe_ratio))
    result.append(np.mean(sortino_ratio))
    result.append(np.mean(calmar_ratio))
    result.append(np.mean(maxdd_ratio))
    result.append(np.mean(treynor_ratio))
    result.append(np.mean(evar_ratio))
    result.append(np.mean(information_ratio))
    result.append(np.mean(upside_ratio))

    return result











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



def eval_results(tick,weights,yearfrom,yearto):

    startdate= str(yearfrom)+'-1-1'
    enddate = str(yearto) + '-12-31'
    middate=str(yearto) + '-1-1'
    df =ff.get_prices(tick,startdate,middate)
    df2= ff.get_prices(tick,middate,enddate)



    X = df[1].as_matrix().reshape(len(df[1]), 1)
    X2 =df2[1].as_matrix().reshape(len(df2[1]), 1)
    y = []
    start_price = []
    #.ravel()
    for r in range(df[0].shape[1]):
        start_price.append(df[0]['open'].iloc[0, r])
    price = prices_change(df,weights,start_price)
    price2= prices_change(df2,weights,start_price)
    plt.plot(X, price, color='darkgreen', label='data')
    plt.plot(X2, price2, color='red', label='data')

    plt.xlabel('Data')
    plt.ylabel('Wartość w procentach')
    plt.title('Zmiana Wartości portfolio')
    plt.show()





def eval_results2(tick,weights,yearfrom,yearto):

    startdate= str(yearfrom)#+'-1-1'
    enddate = str(yearto) #+ '-12-31'
    df =ff.get_prices(tick,startdate,enddate)




    X = df[1].as_matrix().reshape(len(df[1]), 1)
    y = []
    start_price = []
    #.ravel()
    for r in range(df[0].shape[1]):
        start_price.append(df[0]['open'].iloc[0, r])
    price = prices_change(df,weights,start_price)
    plt.plot(X, price, color='darkgreen', label='data')

    plt.xlabel('Data')
    plt.ylabel('Wartość w procentach')
    plt.title('Zmiana Wartości portfolio')
    plt.show()


def eval_results3(tick,tick2,yearfrom,yearto):

    startdate= str(yearfrom)
    enddate = str(yearto)
    df =ff.get_prices(tick,startdate,enddate)
    df2 = ff.get_prices(tick2, startdate, enddate)




    X = df[1].as_matrix().reshape(len(df[1]), 1)
    y = []
    start_price = []
    start_price2 = []
    #.ravel()
    for r in range(df[0].shape[1]):
        start_price.append(df[0]['open'].iloc[0, r])
    for r in range(df2[0].shape[1]):
        start_price2.append(df2[0]['open'].iloc[0, r])
    price = prices_change(df,[1.0],start_price)
    price2 = prices_change(df2, [1.0], start_price2)
    plt.plot(X, price, color='darkgreen', label='data')
    plt.plot(X, price2, color='red', label='data')
    plt.xlabel('Data')
    plt.ylabel('Wartość w procentach')
    plt.title('Zmiana Wartości portfolio')
    plt.show()



def prices_change (df,weights,start_price):
    y = []
    # .ravel()
    for r in range(df[0].shape[1]):
        y.append(df[0]['open'].iloc[:, r].as_matrix().reshape(df[0].shape[0], 1).ravel())
    price = []
    for i in range(len(y[0])):
        avg = []
        for j in range(len(weights)):
            avg.append((y[j][i] / start_price[j]) * 100 * weights[j])
        price.append(sum(avg))
    return (price)


