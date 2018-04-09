import Finance as ff
from scipy.optimize import minimize,fmin
import sys
import tempfile
from amplpy import AMPL
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
    for i in range(0, returns.shape[1]):
        rr = returns.iloc[:, i]
        omega = ff.omega_ratio(rr.as_matrix()) * weights[i] + omega
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
        print'xxx'
        sharpe = ff.calmar_ratio(rr.as_matrix()) * weights[i] + sharpe

    return -1 * sharpe


def optimize(ratio='omega',method='SLSQP',weights =None):
    global returns
    if weights is None:
        weights = []
        for i in range(0,returns.shape[1]):
            weights.append(float(1)/returns.shape[1])

    con = {'type': 'eq', 'fun': cons}
    sol=[]
    bnds = []
    for w in weights:
        bnds.append((0.01, .6))
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
    file.write(b'param {0} := {1};'.format(name, param))


def print_1d_param( file, param, name):

	file.write(b'param {0} :=\n'.format(name))
	for idx, elem in enumerate(param):
		if isinstance(elem, basestring):
				file.write(b'{0} "{1}"'.format(idx + 1, elem))
		else:
				file.write(b'{0} {1}'.format(idx + 1, elem))
		if idx == (param.shape[0] - 1):
			file.write(b';')

		file.write(b' \n ')


def print_2d_param( file, param, name):

		# Write header
	file.write(b'param {0} : '.format(name))
	for i in range(0, param.shape[1]):
		file.write(b'{0} '.format(i + 1))
	file.write(b':=\n')

		# Write data rows
	for idx, row in enumerate(param):
		file.write(b'{0} '.format(idx + 1))
		for elem in row:
			file.write(b'{0} '.format(elem))
		if idx == (param.shape[0] - 1):
			file.write(b';')
		file.write(b'\n')


def write_data_to_temp_file( tmp_file, data):
    print_2d_param(tmp_file, data.demand, 'ZWROTY')
    print_1d_param(tmp_file, data.prices, 'SREDNIE')
    print_1d_param(tmp_file, data.prices, 'PRAWDOPODOBIENSTWA')
    print_scalar_param(tmp_file, 1, 'OKRESY')
    print_scalar_param(tmp_file, 1, 'FIRMY')





def generate_temp_data_file( data):
    tmp_file = tempfile.NamedTemporaryFile()
    write_data_to_temp_file(tmp_file, data)
    print('\nDATA FILE:')
    write_data_to_temp_file(sys.stdout, data)
    tmp_file.seek(0)
    return tmp_file


def run_ampl_model(data):
    dir = os.path.dirname(__file__)
    dir = dir + '\omega.mod'
    ampl = AMPL()
    ampl.eval('option solver Minos;')
    ampl.read(dir)
    data_file = generate_temp_data_file(data)
    ampl.readData(data_file.name)
    data_file.close()
    ampl.solve()
    return ampl





'''
param T := 3 ;
param rf = 0.03 ;
param R = 3 ; 
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
data;
 param u := 1 0.1 2 0.1 3 0.1 ;
 
 param p := 1 0.34  2 0.33 3 0.33 ;
 
 param r: 1  2  3 :=
      1  0.1 0.1 0.1
      2  0.1 0.1 0.1
      3  0.1 0.1 0.1 ;


    
'''




























