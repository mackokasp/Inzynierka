import pandas as pd

import Finance as ff
import GUI as gi
import Graph as gg
import Optimizer as opt

# ticks = sorted(['S', 'WMT', 'VZ', 'RHT', 'DIS', 'JPM', 'BA', 'AMZN', 'PEP', 'ORCL', 'DAL'])

'''
ticks = sorted(['S', 'AAN', 'VZ', 'RHT', 'DIS', 'JPM', 'AAPL', 'F', 'PEP', 'ORCL', 'GE'])

r=ff.month_returns(ticks,2010,2017)
opt.set_returns(r)
ff.set_target(0.005)
sol=opt.optimize(ratio='omega',method='lin_lmt',minW=0.05,maxW=1.0,K=3)
#gg.draw_portfolios_omega(r,ticks,sol)
ff.set_target(0.01)
sol2=opt.optimize(ratio='omega',method='lin_lmt',minW=0.00,maxW=1.0,K=3)
#gg.draw_portfolios_omega(r,ticks,sol2)
ff.set_target(0.015)
sol3=opt.optimize(ratio='omega',method='lin',minW=0.00,maxW=1.0)
#gg.draw_portfolios_omega(r,ticks,sol3)
ff.set_target(0.01)
sol4=opt.optimize(ratio='omega',method='elin',minW=0.00,maxW=1.0)
#gg.draw_portfolios_omega(r,ticks,sol4)


gg.compare_table(r,sol,sol2,sol3,sol4)
'''
gi.start()

tick = sorted(['DAL', 'DIS', 'F'])

ff.set_target(0.01)

r = ff.month_returns(tick, 2012, 2012)
rr = ff.get_prices()
#pred.prediction(rr['date'],rr['open'])

#w=ff.portfolio_omega2(r,[0.2,0.2,0.2,0.2,0.2],target=0.01)
d = {'DAL': [0.3041, -0.0701, 0.0112, 0.1048, 0.1040, -0.0950, -0.1187, -0.1036, 0.059, 0.0513, 0.0384, 0.1870],
     'DIS': [0.0373, 0.0794, 0.0426, -0.0153, 0.0603, 0.0610, 0.0132, 0.0067, 0.0568, -0.0604, 0.01100, 0.0179],
     'F': [0.1590, -0.0032, 0.0081, -0.0921, -0.0638, -0.0919, -0.0365, 0.0164, 0.0557, 0.1369, 0.026, 0.1310]}
df = pd.DataFrame(data=d)

print(df)

r = df




opt.set_returns(r)

# sol =opt.optimize(ratio='omega',method='SLSQP')

sol = opt.optimize(ratio='omega', method='lin', minW=0.00, maxW=1.0)


#gg.eval_results3(tick,['CHD'] ,'2014-01-01','2017-06-15')
#gg.eval_results2(tick,sol ,'2013-04-01','2015-01-15')

gg.draw_portfolios_omega(r,tick,sol)






'''
opt.set_returns(r)
ff.set_target(0.3)
sol2 = opt.optimize(ratio='omega',method='lin',maxW=1.0)
gg.draw_portfolios_omega(r,tick,sol2)
gg.draw_table(r,sol2)





r2=ff.daily_returns(tickers)


r = ff.daily_returns(tick,dateto='2017-12-31')
opt.set_returns(r)



sol = opt.optimize(ratio='omega',method='lin')
opt.set_returns(r2)
sol3 = opt.optimize(ratio='omega',method='SLSQP')

#gg.draw_table(r2,sol3)
gg.draw_portfolios_omega(r,tick,sol)

gg.draw_portfolios_omega(r2,tickers,sol3)
'''
'''
sols= opt.optimize(ratio='sharpe',method='fmin')



rr= ff.get_prices()
pred.prediction(rr['date'],rr['open'])




#ff.benchmark_data()
#
#weights=[0.2,0.2,0.2,0.2,0.2,0.2]
#r = ff.daily_returns(tick)
#opt.set_returns(r)
#sol =(opt.optimize(weights,ratio='calmar',method='Nelder-Mead'))
#print sol
#print ff.portfolio_omega(r,sol)
#gg.draw_portfolios_calmar(r,tick,sol)
'''

'''
sol2=(opt.optimize(weights,metfhod='SLSQP' ))
sol3=(opt.optimize(weights,method='Nelder-Mead'))

print (opt.normalize(sol))
print opt.normalize(sol2)
print opt.normalize(sol3)

print  opt.omega_opt(opt.normalize(sol3))
print  opt.omega_opt(opt.normalize(sol2))

print  opt._opt(opt.normalize(sol))
def randomize (metadata,yearfrom, yearto, num):
    tickers=[]
    yearfrom_date = pd.to_datetime(str(yearfrom) + '0101', format='%Y%m%d', errors='coerce')
    yearto_date = pd.to_datetime(str(yearto) + '1231', format='%Y%m%d', errors='coerce')

    for i in range (metadata.shape[0]):

        st_eod   = pd.to_datetime(metadata.iloc[i,4], format='%Y-%m-%d', errors='coerce')

        end_eod = pd.to_datetime(metadata.iloc[i,5], format='%Y-%m-%d', errors='coerce')

        if  st_eod < yearfrom_date and end_eod > yearto_date:
            tickers.append(metadata.iloc[i,0])
            if len(tickers) > 3 :
                break ;
    return tickers

'''
