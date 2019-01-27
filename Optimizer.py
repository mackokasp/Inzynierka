
import os

from scipy.optimize import minimize, fmin

import AMPL as amp
import Finance as ff

returns= []
def set_returns (ret):
    global returns
    returns = ret


def cons(weights):
    return sum(weights) - 1


def normalize(weights):
    for i in range(0, len(weights) ):
        weights[i]=abs(weights[i])
    suma=sum(weights)
    nweights = weights
    for i in range(0, len(weights)):
        nweights[i]=nweights[i]/suma
    return (nweights )


def omega_opt( weights):
    global returns
    omega=0
    weights=normalize(weights)

    omega = ff.portfolio_omega(returns, weights)
    return -1 * omega


def sharpe_opt( weights):
    global returns
    weights=normalize(weights)
    sharpe = 0
    for i in range(0, returns.shape[1]):
        rr = returns.iloc[:, i]
        sharpe = ff.sharpe_ratio(rr.as_matrix()) * weights[i] + sharpe
    return -1 * sharpe


def calmar_opt (weights):
    global returns
    weights = normalize(weights)
    sharpe = 0
    for i in range(0, returns.shape[1] ):
        rr = returns.iloc[:, i]
        sharpe = ff.calmar_ratio(rr.as_matrix()) * weights[i] + sharpe

    return -1 * sharpe


def optimize(ratio='omega', method='SLSQP', minW=0.01, maxW=0.8, weights=None, K=3):
    global returns
    os.environ['PATH'] = 'C:\\Users\\PC\INŻ\\ampl4'+ '|' + os.environ['PATH']
    if weights is None:
        weights = []
        for i in range(0,returns.shape[1]):
            weights.append(float(1)/returns.shape[1])

    con = {'type': 'eq', 'fun': cons}
    sol=[]
    bnds = []
    for w in weights:
        bnds.append((minW, maxW))
    if ratio=='omega':
        if method =='SLSQP':
            # sol = minimize(omega_opt, method='SLSQP', x0=weights,bounds=bnds, constraints=con,tol=0.0001)
            sol = minimize(omega_opt, method='Nelder-Mead', x0=weights, tol=0.000001)


        elif method=='Nelder-Mead':
            sol = minimize(omega_opt, method='Nelder-Mead', x0=weights)
        elif method=='Newton-CG':
            sol = minimize(omega_opt, method='Newton-CG', x0=weights,jac=False)
        elif method=='fmin' :
            sol = fmin(omega_opt, x0=weights)
            return sol
        elif method =='lin':
            sol = amp.run_ampl_model(returns, minW=minW, maxW=maxW, ex=0)
            return sol
        elif method =='elin':
            sol = amp.run_ampl_model(returns, minW=minW, maxW=maxW, ex=1)
            return sol
        elif method == 'lin_lmt':
            sol = run_ampl_model(returns, minW=minW, maxW=maxW, ex=2, K=K)
            return sol
        elif method == 'lin_safe':
            sol = amp.run_ampl_model(returns, minW=minW, maxW=maxW, ex=3, K=K)
            return sol
        else:
            sol = minimize(omega_opt, method=method, x0=weights)


    elif ratio=='sharpe':
        sol= fmin(sharpe_opt, x0=weights)
        return sol

    elif ratio=='calmar' :
        sol = sol = minimize(calmar_opt, method='SLSQP', x0=weights,bounds=bnds, constraints=con,options={'maxiter':200},tol=0.1)



    else:


        return sol

    return  sol.x







'''
param T  ;
param rf; 
param R  ; 
param  u{1..R} ;
param r{1..R,1..T} ;
param p{1..T };
var x{1..R};
var v0 ;
var z ;
var v ;
var d{1..T};
var y{1..T} ;
maximize fun: v + rf * v0 ;

subject to
 c1: sum{i in 1..R}x[i]= v0;
 
 c2: sum{i in 1..R}x[i]*u[i]=v ;
 
 c3{t in 1..T}: sum{i in 1..R}r[i,t]*x[i]=y[t] ;
 
 c4: sum{t in 1..T}d[t]*p[t]=1 ;
 
 c5{t in 1..T}: d[t] >= rf*v0 - y[t];
 
 c6{t in 1..T}: d[t] >= 0 ;
 
 c7: v0 <= 10000 ;
 
 c8{i in 1..R}: x[i]>=0 ;


    
'''




























