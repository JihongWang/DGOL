import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
 

from scipy.interpolate import griddata
 

import time, argparse, os
from WaveData_utilities import *


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
    u0_fun_test, ut0_fun_test, f_fun_test, _, _ = test_data()

    domain = config['data']['domain']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']
    Nx_u, Nx_f, Nt_f = config['data']['Nx_u'], config['data']['Nx_f'], config['data']['Nt_f']
    x_u = np.linspace(xmin, xmax, Nx_u)
    x_f = np.linspace(xmin, xmax, Nx_f)
    t_f = np.linspace(tmin, tmax, Nt_f)
    X_f, T_f = np.meshgrid(x_f, t_f)
    x_f, t_f = X_f.reshape(1,-1), T_f.reshape(1,-1)

    u0_test, ut0_test, f_test = u0_fun_test(x_u), ut0_fun_test(x_u), f_fun_test(x_f,t_f).flatten()
    u0_max, ut0_max, f_max = np.max(np.abs(u0_test)), np.max(np.abs(ut0_test)), np.max(np.abs(f_test))
    print('u0_max:%f, ut0_max:%f, f_max:%f' % (u0_max, ut0_max, f_max))


    tol = config['data']['tol']
    Num = config['data']['N_input']

    index = 0
    index_outer = 0
    Coe = np.zeros((7, Num))
    while index < Num:
        index_outer +=1

        coe = np.zeros((7,))
        coe[0] = np.random.normal(0, 1, (1,))
        coe[1] = np.random.normal(0, 1, (1,))
        coe[2] = np.random.normal(0, 1, (1,))
        coe[3] = np.random.normal(1, 1, (1,))
        coe[4] = np.random.normal(1, 1, (1,))
        coe[5] = np.random.normal(1, 1, (1,))
        coe[6] = np.random.normal(-1, 1, (1,))
        u0_fun_train, ut0_fun_train, f_fun_train, u_fun_train = train_data(coe)
        u0_train, ut0_train, f_train, u_train = u0_fun_train(x_u), ut0_fun_train(x_u), f_fun_train(x_f,t_f), u_fun_train(x_f,t_f)

        u0_err = np.linalg.norm(u0_test - u0_train)/np.linalg.norm(u0_test)
        ut0_err = np.linalg.norm(ut0_test - ut0_train)/np.linalg.norm(ut0_test)
        f_err = np.linalg.norm(f_test - f_train)/np.linalg.norm(f_test)
        u0_max, ut0_max, f_max = np.max(u0_train), np.max(ut0_train), np.max(f_train)

        
        if (u0_err <= tol['u0'] and ut0_err <= tol['ut0'] and f_err <= tol['f']) and u0_max >=0.7 and f_max >= 0.9:
            Coe[:,index]= coe
            print(index)
            index += 1

    print(index_outer)

    save_path = '%s/data_%s_%s_%s_%s.mat' % (dir, Num, tol['u0'], tol['ut0'], tol['f'])
    sio.savemat(save_path, {'Coe':Coe})
    


if __name__ == "__main__":
    main()