import Finance as ff
import Optimizer as opt
import Graph as gg
import numpy as numpy
import numpy.random as nrand
import pandas
import os
import Prediction as pred


tick = ['CNP', 'F', 'WMT', 'GE','AAPL']
weights=[0.2,0.2,0.1,0.1,0.2]
r = ff.daily_returns(tick)
opt.set_returns(r)
sol = opt.optimize(weights,ratio='omega',method='fmin')
sol2= opt.optimize(weights,ratio='omega',method='SLSQP')
ff.set_target(0.0)
sol3 = opt.optimize(weights,ratio='omega',method='Nelder-Mead')
gg.draw_portfolios_omega(r,tick,sol)
gg.draw_portfolios_omega(r,tick,sol2)
gg.draw_portfolios_omega(r,tick,sol3)

sols= opt.optimize(weights,ratio='sharpe',method='fmin')
gg.draw_portfolios_sharpe(r,tick,sols)


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
sol2=(opt.optimize(weights,metfhod='SLSQP' ))
sol3=(opt.optimize(weights,method='Nelder-Mead'))

print (opt.normalize(sol))
print opt.normalize(sol2)
print opt.normalize(sol3)

print  opt.omega_opt(opt.normalize(sol3))
print  opt.omega_opt(opt.normalize(sol2))

print  opt._opt(opt.normalize(sol))
'''

