import Finance as ff
import Optimizer as opt
import Graph as gg
import django
import GUI as gi

import numpy as numpy
import numpy.random as nrand
import pandas as pd
import os
import Prediction as pred

gi.start()
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

eod = pd.read_csv('EOD_metadata.csv')


#print (eod.shape[0])









#pred.prediction(rr['date'],rr['open'])


#gi.start()

tick = sorted(['AIR','AAN', 'F', 'GE','WMT'])
tickers=sorted (['AAN','GM','AIR','BA','CNP', 'GE' ,'V'] )
#gg.eval_results4(tick,2011,2012)
ff.set_target(0.01)
tick =randomize(eod,2013,2013,0)
print(tick)
r =ff.month_returns(tick,2013,2013)
#w=ff.portfolio_omega2(r,[0.2,0.2,0.2,0.2,0.2],target=0.01)



opt.set_returns(r)

sol =opt.optimize(ratio='omega',method='SLSQP')

sol=opt.optimize(ratio='omega',method='lin',minW=0.01,maxW=1.0)


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
'''

