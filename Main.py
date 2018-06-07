import Finance as ff
import Optimizer as opt
import Graph as gg

import GUI as gi

import numpy as numpy
import numpy.random as nrand
import pandas
import os
import Prediction as pred
import Sentiment as st
st.main()
tick = sorted(['AAN','CNP', 'F', 'GE' ])
gg.eval_results(tick,[0.5,0.5,0.0,0.0],2000,2017)
#pred.prediction(rr['date'],rr['open'])
#gi.start()
'''
tick = sorted(['AAPL','CNP', 'F', 'GE' ])
tickers=sorted (['AAN','GM','AIR','BA','CNP', 'GE' ] )
ff.set_target(0.3)
r =ff.year_returns(tick,2010,2017)
opt.set_returns(r)
sol =opt.optimize(ratio='omega',method='SLSQP')

gg.draw_portfolios_omega(r,tick,sol)
'''
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
