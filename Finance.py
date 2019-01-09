import math
import os

import numpy as np
import pandas as pd
import quandl

quandl.ApiConfig.api_key = 'xux8UTaML7UgmBhuGy4v'

benchmark = []

gtarget= None

def set_target(targ):
    global gtarget
    gtarget = targ

def set_average(returns):
     av=np.mean(returns)
     a1 = np.mean(av)
     global gtarget
     gtarget=a1


def benchmark_data():
     global benchmark
     dir = os.path.dirname(__file__)
     dir = dir + '\sp500.csv'

     df =pd.read_csv(dir)
     df=df.dropna()
     benchmark = df

     return df

def get_prices(ticker,dfrom ='2015-1-1',dto='2018-1-10'):
        data = quandl.get_table('WIKI/PRICES', ticker=ticker,
                                qopts={'columns': ['date', 'Open','ticker']},
                                date={'gte': dfrom, 'lte': dto}, paginate=True)
        data2 = quandl.get_table('WIKI/PRICES', ticker=ticker[0],
                                qopts={'columns': ['date', 'Open', 'ticker']},
                                date={'gte': dfrom, 'lte': dto}, paginate=True)
        dates= data2['date'].sort_values()

        res=[]
        clean = data.set_index('date')
        table = clean.pivot(columns='ticker')
        returns_daily = table
        data = returns_daily

        data = data.dropna(axis=0)
        data.fillna(method='ffill', inplace=True)




        data.sort_index(inplace=True)


        res.append(data)
        res.append(dates)
        return res




def graph_data(tickers):
    data = quandl.get_table('WIKI/PRICES', ticker=tickers,
                            qopts={'columns': ['date', 'ticker', 'Open']},
                            date={'gte': '2015-1-1', 'lte': '2017-12-31'}, paginate=True)
    return data


def daily_returns (tickers,datefrom='2015-1-1',dateto ='2017-12-31'):
    data = quandl.get_table('WIKI/PRICES', ticker = tickers,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': datefrom, 'lte': dateto }, paginate=True)

    clean = data.set_index('date')
    table = clean.pivot(columns='ticker')
    returns_daily = table.pct_change()
    data=returns_daily
    data = data.dropna()
    global daily_return
    daily_return= data
    return data


def make_date (year,month,day):
    return str(year)+ '-' + str(month).zfill(2)+'-'+str(day)


def year_returns (tickers,yearfrom ,yearto ):
    frets =[]
    if 1==1:
        datefrom = make_date(yearfrom, 1, 1)
        dateto = make_date(yearto, 12, 31)
        data = quandl.get_table('WIKI/PRICES', ticker=tickers,
                                qopts={'columns': ['date', 'ticker', 'adj_close']},
                                date={'gte': datefrom, 'lte': dateto}, paginate=True)
        data= data.dropna()
        data = data.pivot(index='date', columns='ticker', values='adj_close')
        data=data.dropna(axis=0,how='any')

        for i in range(yearto+1-yearfrom):

            sttime=pd.to_datetime(str(yearfrom+i)+ '0101', format='%Y%m%d', errors='coerce')
            entime =pd.to_datetime(str(yearfrom+i)+ '1231', format='%Y%m%d', errors='coerce')
            data2=data.loc[sttime:entime]
            rets = []
            for j in range(data.shape[1]):
                startprice = data2.iloc[0, j]
                endprice = data2.iloc[-1, j]
                ret = (endprice - startprice) / startprice
                rets.append(ret)

            frets.append(rets)




    df = pd.DataFrame(frets,columns=tickers)

    return df


def month_returns (tickers,yearfrom ,yearto ):
    days=['31','28','31','30','31','30','31','31','30','31','30','31']
    frets =[]
    if 1==1:
        datefrom = make_date(yearfrom, 1, 1)
        dateto = make_date(yearto, 12, 31)
        data = quandl.get_table('WIKI/PRICES', ticker=tickers,
                                qopts={'columns': ['date', 'ticker', 'adj_close']},
                                date={'gte': datefrom, 'lte': dateto}, paginate=True)
        data= data.dropna()
        data = data.pivot(index='date', columns='ticker', values='adj_close')
        data=data.dropna(axis=0,how='any')

        for i in range(yearto+1-yearfrom):

            for m in range(1,13 ) :
                sttime=pd.to_datetime(str(yearfrom+i) + str(m).zfill(2)+'01', format='%Y%m%d', errors='coerce')
                entime =pd.to_datetime(str(yearfrom+i)+  str(m).zfill(2)+days[m-1], format='%Y%m%d', errors='coerce')
                data2=data.loc[sttime:entime]

                rets = []
                for j in range(data.shape[1]):
                    startprice = data2.iloc[0, j]
                    endprice = data2.iloc[-1, j]
                    ret = (endprice - startprice) / startprice
                    rets.append(ret)


                frets.append(rets)







    df = pd.DataFrame(frets,columns=tickers)
    return df












def get_return(ticker ,dateFrom,dateTo):
    startPrice= quandl.get('WIKI/'+ticker,start_date=dateFrom+'-01' ,end_date=dateFrom+'-05')['Open'].iloc[0]
    endPrice = quandl.get('WIKI/'+ticker, start_date=dateTo + '-01', end_date=dateTo + '-05')['Open'].iloc[0]
    return ( (endPrice-startPrice)-startPrice)/startPrice


def vol(returns):
    return np.std(returns)


def lpm(returns, threshold, order):
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    diff = threshold_array - returns
    diff = diff.clip(min=0)
    db= np.sum(diff ** order) / len(returns)
    return db

def hpm(returns, threshold, order):
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    diff = returns - threshold_array
    diff = diff.clip(min=0)
    return np.sum(diff ** order) / len(returns)


def var(returns, alpha):
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    return abs(sorted_returns[index])


def cvar(returns, alpha):
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    return abs(sum_var / index)


def excess_var(er, returns, rf=0.0, target=0.0):
    if gtarget is not None:
        target = gtarget
        rf=target

    return (er - rf) / var(returns, target)


def information_ratio(returns, benchmark):
    diff = returns - benchmark
    return np.mean(diff) / vol(diff)


def treynor_ratio(er, returns, market, rf=0.0):
    if gtarget is not None:
        rf = gtarget
    return (er - rf) / beta(returns, market)





def average_dd(returns, periods):
    # Returns the average maximum drawdown over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def upside_potential_ratio(returns, target=0):
    if gtarget is not None:
        target = gtarget
    return hpm(returns, target, 1) / math.sqrt(lpm(returns, target, 2))

def beta(returns, market):
    # Create a matrix of [returns, market]
    m = np.matrix([returns, market])
    # Return the covariance of m divided by the standard deviation of the market returns
    return np.cov(m)[0][1] / np.std(market)


def sharpe_ratio( returns, rf=0.00):
    if gtarget is not None:
        target = gtarget
        rf=target
    er = np.mean(returns)
    return (er - rf) / vol(returns)


# er= mean , rf = 0.06
def omega_ratio(returns, rf=0.03, target=0.05):


    global gtarget
    if gtarget is not None:
        target= gtarget
        rf=target
    er = np.mean(returns)
    omega =(er - rf) / lpm(returns, target, 1)
    return omega


def omega2(returns, target =0.0):
    upside = 0.0
    downside = 0.0
    if gtarget is not None:
        target = gtarget
    for rr in returns:

        if rr > target:
            upside = upside + (rr-target)
        else:
            downside= downside + (target-rr)

    omega2= upside/downside
    return omega2





def sortino_ratio( returns, rf=0.03, target=0):
    if gtarget is not None:
        target = gtarget
        rf=target
    er = np.mean(returns)
    return (er - rf) / math.sqrt(lpm(returns, target, 1))

def portfolio_omega(returns,weights,rf=0.03,target =0.1):
    global gtarget
    if gtarget is not None:
        target = gtarget
        rf = target
    omega = 0
    weights = normalize(weights)
    rets =0
    for i in range(0, returns.shape[1]):
        rr = returns.iloc[:, i] * weights[i]
        rr=rr.as_matrix()
        #print (rr)
        rets=rets+rr

    omega = omega_ratio(rets, target=target)

    return  omega

def portfolio_omega2(returns,weights,rf=0.03,target =0.1):
    global gtarget
    if gtarget is not None:
        target = gtarget
        rf = target
    omega = 0
    weights = normalize(weights)
    rets =[]
    for i in range(0, returns.shape[0]):
        rr=0
        for j in range(0,returns.shape[1]):
            rr = returns.iloc[i, j] * weights[j] + rr
        rets.append(rr)

    omega = omega2(rets,target=target)
    return  omega


def portfolio_vol(returns,weights ):
    vola = 0
    weights = normalize(weights)
    for i in range(0, returns.shape[1]):
        rr = returns.iloc[:, i]
        vola = vol(rr.as_matrix()) * weights[i] + vola
    return vola


def normalize(weights):
    for i in range(0, len(weights) ):
        weights[i]=abs(weights[i])
    suma=sum(weights)
    nweights = weights
    for i in range(0, len(weights)):
        nweights[i]=nweights[i]/suma
    return (nweights )

def modigliani_ratio( returns, benchmark, rf):
    if gtarget is not None:
        target = gtarget
        rf=target
    er=np.mean(returns)
    np_rf = np.empty(len(returns))
    np_rf.fill(rf)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (er - rf) * (vol(rdiff) / vol(bdiff)) + rf

def calmar_ratio( returns, rf=0.03):
    if gtarget is not None:
        target = gtarget
        rf=target
    er= returns.mean()
    return (er - rf) / max_dd(returns)

def max_dd(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)

def dd(returns, tau):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)

def prices(returns, base):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return np.array(s)
