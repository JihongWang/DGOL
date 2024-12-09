import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
from scipy.interpolate import griddata
import time, argparse, os
from Wave2D_utilities import *
import torch
import torch.nn as nn
import numpy as np
import yaml, math
from argparse import ArgumentParser
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio

def main():

    parser = argparse.ArgumentParser('')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    dir = os.path.dirname(os.path.realpath(__file__))
    config_file = '%s/%s' % (dir, args.config_path)

    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    
    domain = config['data']['domain']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

 
    Nx_u, Nx_f, Nt_f = config['data']['Nx_u'], config['data']['Nx_f'], config['data']['Nt_f']
    x_u = np.linspace(xmin, xmax, Nx_u)
    X1_u, X2_u = np.meshgrid(x_u, x_u)
    x1_u, x2_u = X1_u.reshape(1,-1), X2_u.reshape(1,-1)
    x_f = np.linspace(xmin, xmax, Nx_f)
    t_f = np.linspace(tmin, tmax, Nt_f)
    X1_f, X2_f, T_f = np.meshgrid(x_f, x_f, t_f)
    x1_f, x2_f, t_f = X1_f.reshape(1,-1), X2_f.reshape(1,-1), T_f.reshape(1,-1)

 
    if_exact_sol = config['data']['if_exact_sol']
    if if_exact_sol == 1:
        u0_fun_test, ut0_fun_test, f_fun_test, u_fun_test = test_data()
 
        u0_test, ut0_test, f_test, u_test = u0_fun_test(x1_u, x2_u), ut0_fun_test(x1_u, x2_u), f_fun_test(x1_f, x2_f, t_f), u_fun_test(x1_f, x2_f, t_f)
    else:
        u0_fun_test, ut0_fun_test, f_fun_test = test_data()
 
        u0_test, ut0_test, f_test = u0_fun_test(x1_u, x2_u), ut0_fun_test(x1_u, x2_u), f_fun_test(x1_f, x2_f, t_f)


    
    u0_max, ut0_max, f_max = np.max(np.abs(u0_test)), np.max(np.abs(ut0_test)), np.max(np.abs(f_test))
    print('u0_max:%f, ut0_max:%f, f_max:%f' % (u0_max, ut0_max, f_max))


    tol = config['data']['tol']
    Num = config['data']['N_input']
    order_x = config['data']['order_x']
    type_x, type_t = config['data']['type_x'], config['data']['type_t']
    Coe = np.zeros((7, Num))

    index = 0
    index_outer = 0
    while index < Num:
        index_outer +=1
        coe = np.zeros((7,))
        coe[0], coe[1], coe[2], coe[3], coe[4], coe[5], coe[6]  = np.random.normal(1, 1, (1,)), np.random.normal(2, 1, (1,)), np.random.normal(1, 1, (1,)), \
            np.random.normal(1, 1, (1,)), np.random.normal(0, 1, (1,)), np.random.normal(0, 1, (1,)), np.random.normal(3, 1, (1,))

        u0_fun_train, ut0_fun_train, f_fun_train,  u_fun_train = train_data(coe)
        u0_train, ut0_train, f_train = u0_fun_train(x1_u, x2_u), ut0_fun_train(x1_u, x2_u), f_fun_train(x1_f, x2_f, t_f)

        u0_err = np.linalg.norm(u0_test - u0_train)/np.linalg.norm(u0_test)
        ut0_err = np.linalg.norm(ut0_test - ut0_train)/np.linalg.norm(ut0_test)
        f_err = np.linalg.norm(f_test - f_train)/np.linalg.norm(f_test)


        if u0_err <= tol['u0'] and  ut0_err <= tol['ut0'] and f_err <= tol['f']:
            Coe[:,index] = coe
            print(index)
            index += 1

    save_path = '%s/data_%s_%s_%s_%s.mat' % (dir, Num, tol['u0'], tol['ut0'], tol['f'])
    sio.savemat(save_path, {'Coe':Coe})
    print(index_outer)



if __name__ == "__main__":
    main()