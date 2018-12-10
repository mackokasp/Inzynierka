import Finance as ff
from scipy.optimize import minimize,fmin
import sys
import numpy as np

from amplpy import AMPL,Environment
#import amplpy
import os
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

    omega = ff.portfolio_omega2(returns,weights)
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


def optimize(ratio='omega',method='SLSQP',minW=0.01,maxW=0.8,weights =None):
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
            sol = minimize(omega_opt, method='SLSQP', x0=weights,bounds=bnds, constraints=con)
        elif method=='Nelder-Mead':
            sol = minimize(omega_opt, method='Nelder-Mead', x0=weights)
        elif method=='Newton-CG':
            sol = minimize(omega_opt, method='Newton-CG', x0=weights,jac=False)
        elif method=='fmin' :
            sol = fmin(omega_opt, x0=weights)
            return sol
        elif method =='lin':
            sol = run_ampl_model(returns,minW=minW,maxW=maxW)
            return sol
        elif method =='elin':
            sol = run_ampl_model(returns,minW=minW,maxW=maxW,ex=1)
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


def print_scalar_param( file, param, name):
    file.write('param {0} := {1};'.format(name, param))


def print_1d_param( file, param, name):
    file.write('param {0} := \n '.format(name))
    for i in range(len(param)):
        file.write('{0} {1:.3f}'.format(i + 1, param[i]))
        if i == (len(param) - 1):
            file.write(';')
        file.write('\n')






def print_2d_param( file, param, name):
    file.write('param {0} : '.format(name))
    for i in range(0, param.shape[1]):
        file.write('{0} '.format(i + 1))
    file.write(':=\n')
    for idx in range(param.shape[0]):
        file.write('{0} '.format(idx + 1))
        for jix in range (param.shape[1]):
            file.write('{0} '.format(param.iloc[idx,jix]))
        if idx == (param.shape[0] - 1):
            file.write(';')
        file.write('\n')


def write_data_to_temp_file( tmp_file, returns ,maxW =0.6,minW=0.01):
    print_2d_param(tmp_file, returns, 'r')
    means=[]
    weights=[]
    for i in range (returns.shape[1]):
       # means.append(np.mean(returns.iloc[:,i])*returns.shape[0])
        means.append(np.mean(returns.iloc[:, i]))
    for i in range(returns.shape[0]):
        weights.append(float(1) / returns.shape[0])
    print_1d_param(tmp_file, means, 'u')
    print_1d_param(tmp_file, weights, 'p')
    print_scalar_param(tmp_file, returns.shape[0], 'T')
    print_scalar_param(tmp_file, ff.gtarget, 'rf')
    print_scalar_param(tmp_file, returns.shape[1], 'R')
    print_scalar_param(tmp_file, maxW, 'maxW')
    print_scalar_param(tmp_file, minW, 'minW')

def write_data_to_temp_file_ex( tmp_file, returns ,maxW =0.6,minW=0.01):
    print_2d_param(tmp_file, returns, 'r')
    means=[]
    weights=[]
    time_means=[]
    for i in range (returns.shape[1]):
        #means.append(np.mean(returns.iloc[:,i])*returns.shape[0])
        means.append(np.mean(returns.iloc[:, i]) )
    for i in range(returns.shape[0]):
        weights.append(float(1) / returns.shape[0])

    for i in range(returns.shape[0]):
        time_means.append(np.mean(returns.iloc[i,:])*returns.shape[1])
        #print (returns.iloc[1,:])



    print_1d_param(tmp_file, means, 'u')
    print_1d_param(tmp_file, weights, 'p')
    print_1d_param(tmp_file, time_means, 'ra')
    print_scalar_param(tmp_file, returns.shape[0], 'T')
    print_scalar_param(tmp_file, ff.gtarget, 'rf')
    print_scalar_param(tmp_file, returns.shape[1], 'R')
    print_scalar_param(tmp_file, maxW, 'maxW')
    print_scalar_param(tmp_file, minW, 'minW')





def generate_temp_data_file( data,minW=0.01,maxW=0.6):
    tmp_file = open('data.dat', 'w')
    write_data_to_temp_file(tmp_file, data,maxW,minW)
    tmp_file.seek(0)
    return tmp_file

def generate_temp_data_file_ex( data,minW=0.01,maxW=0.6):
    tmp_file = open('data.dat', 'w')
    write_data_to_temp_file_ex(tmp_file, data,maxW,minW)
    tmp_file.seek(0)
    return tmp_file


def run_ampl_model(data,minW=0.01,maxW=0.6,ex=0):
    dir = os.path.dirname(__file__)
    dir='C:\\Biblioteki\\AMPL\\ampl.mswin64'
    #dir=dir.replace('/','\\')


    #ampl = AMPL()
    ampl = AMPL(Environment(dir))

    #asp
    ampl.setOption('solver',dir+'\minos.exe')
    if ex<1:
        ampl.read('omg3.txt')
        data_file = generate_temp_data_file(data, minW=minW, maxW=maxW)
    else:
        ampl.read('eomg.txt')
        data_file = generate_temp_data_file_ex(data, minW=minW, maxW=maxW)



    #data_file=open('data_pattern.txt')
    ampl.readData(data_file.name)
    data_file.close()
    ampl.solve()
    x = get_np_array_from_variable(ampl, 'x')
    #x=ampl.getVariable('x').getValues().toPandas()



    v0=ampl.getVariable('v0').getValues().toPandas()

    sol = []

    for i in range(x.shape[1]):
        sol.append(x.iloc[0,i]/v0.iloc[0,0])


    return sol





def get_np_array_from_variable( ampl, name):
    tmp = np.transpose(ampl.getVariable(name).getValues().toPandas())
    return tmp





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




























