import os

import numpy as np
from amplpy import AMPL, Environment

import Finance as ff


def print_scalar_param(file, param, name):
    file.write('param {0} := {1};'.format(name, param))


def print_1d_param(file, param, name):
    file.write('param {0} := \n '.format(name))
    for i in range(len(param)):
        file.write('{0} {1:.5f}'.format(i + 1, param[i]))
        if i == (len(param) - 1):
            file.write(';')
        file.write('\n')


def print_2d_param(file, param, name):
    file.write('param {0} : '.format(name))
    for i in range(0, param.shape[1]):
        file.write('{0} '.format(i + 1))
    file.write(':=\n')
    for idx in range(param.shape[0]):
        file.write('{0} '.format(idx + 1))
        for jix in range(param.shape[1]):
            file.write('{0} '.format(param.iloc[idx, jix]))
        if idx == (param.shape[0] - 1):
            file.write(';')
        file.write('\n')


def write_data_to_temp_file(tmp_file, returns, maxW=0.6, minW=0.01):
    print_2d_param(tmp_file, returns, 'r')
    means = []
    weights = []
    for i in range(returns.shape[1]):
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


def write_data_to_temp_file_ex(tmp_file, returns, maxW=0.6, minW=0.01):
    print_2d_param(tmp_file, returns, 'r')
    means = []
    weights = []
    time_means = []
    for i in range(returns.shape[1]):
        # means.append(np.mean(returns.iloc[:,i])*returns.shape[0])
        means.append(np.mean(returns.iloc[:, i]))
    for i in range(returns.shape[0]):
        weights.append(float(1) / returns.shape[0])

    for i in range(returns.shape[0]):
        time_means.append(np.mean(returns.iloc[i, :]) * returns.shape[1])
        # print (returns.iloc[1,:])

    print_1d_param(tmp_file, means, 'u')
    print_1d_param(tmp_file, weights, 'p')
    print_1d_param(tmp_file, time_means, 'ra')
    print_scalar_param(tmp_file, returns.shape[0], 'T')
    print_scalar_param(tmp_file, ff.gtarget, 'rf')
    print_scalar_param(tmp_file, returns.shape[1], 'R')
    print_scalar_param(tmp_file, maxW, 'maxW')
    print_scalar_param(tmp_file, minW, 'minW')


def write_data_to_temp_file_lmt(tmp_file, returns, maxW=0.6, minW=0.01, K=3):
    print_2d_param(tmp_file, returns, 'r')
    means = []
    weights = []
    for i in range(returns.shape[1]):
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
    print_scalar_param(tmp_file, K, 'K');


def generate_temp_data_file(data, minW=0.01, maxW=0.6):
    tmp_file = open('data.dat', 'w')
    write_data_to_temp_file(tmp_file, data, maxW, minW)
    tmp_file.seek(0)
    return tmp_file


def generate_temp_data_file_ex(data, minW=0.01, maxW=0.6):
    tmp_file = open('data.dat', 'w')
    write_data_to_temp_file_ex(tmp_file, data, maxW, minW)
    tmp_file.seek(0)
    return tmp_file


def generate_temp_data_file_lmt(data, minW=0.01, maxW=0.6, K=3):
    tmp_file = open('data.dat', 'w')
    write_data_to_temp_file_lmt(tmp_file, data, maxW, minW, K=K)
    tmp_file.seek(0)
    return tmp_file


def run_ampl_model(data, minW=0.01, maxW=0.6, ex=0, K=3):
    dir = os.path.dirname(__file__)
    dir = 'C:\\Biblioteki\\AMPL\\ampl.mswin64'
    # dir=dir.replace('/','\\')

    # ampl = AMPL()
    ampl = AMPL(Environment(dir))

    if ex < 1:
        ampl.read('omg3.txt')
        data_file = generate_temp_data_file(data, minW=minW, maxW=maxW)
        ampl.setOption('solver', dir + '\minos.exe')
    elif ex == 2:

        ampl.read('omg_lmt.txt')

        data_file = generate_temp_data_file_lmt(data, minW=minW, maxW=maxW, K=K)
        ampl.setOption('solver', dir + '\cplex.exe')

    elif ex == 3:
        # ampl.read('omg_lmt.txt')
        ampl.read('omg4.txt')

        data_file = generate_temp_data_file_lmt(data, minW=minW, maxW=maxW, K=K)
        ampl.setOption('solver', dir + '\cplex.exe')

    else:
        ampl.read('eomg.txt')
        data_file = generate_temp_data_file_ex(data, minW=minW, maxW=maxW)
        ampl.setOption('solver', dir + '\minos.exe')

    # data_file=open('data_pattern.txt')
    ampl.readData(data_file.name)
    # ampl.readData('data_test.dat')
    data_file.close()
    ampl.solve()
    x = get_np_array_from_variable(ampl, 'x')
    # x=ampl.getVariable('x').getValues().toPandas()

    v0 = ampl.getVariable('v0').getValues().toPandas()

    sol = []

    for i in range(x.shape[1]):
        sol.append(x.iloc[0, i] / v0.iloc[0, 0])

    return sol


def get_np_array_from_variable(ampl, name):
    tmp = np.transpose(ampl.getVariable(name).getValues().toPandas())
    return tmp
