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

 

    domain = config['data']['domain']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']
 
    order_x = config['data']['order_x']
    type_x, type_t = config['data']['type_x'], config['data']['type_t']
    N = config['data']['N_input']

    Nx_u, Nx_f, Nt_f = config['data']['Nx_u'], config['data']['Nx_f'], config['data']['Nt_f']
    Nx_f, Nt_f = 101, 101
    x_u = np.linspace(xmin, xmax, Nx_u)
    x_f = np.linspace(xmin, xmax, Nx_f)
    t_f = np.linspace(tmin, tmax, Nt_f)
    X_f, T_f = np.meshgrid(x_f, t_f)
    x_f, t_f = X_f.reshape(1,-1), T_f.reshape(1,-1)
    Nf = Nx_f*Nt_f
    u0_train, ut0_train, f_train = np.zeros((N,Nx_u)), np.zeros((N, Nx_u)), np.zeros((N,Nf))


 
    u0_fun_test, ut0_fun_test, f_fun_test, u_fun_test = test_data()
    f_test, u_test = f_fun_test(x_f,t_f), u_fun_test(x_f,t_f)

    plt.figure()
 
    plt.pcolor(X_f, T_f, f_test.reshape(Nx_f, Nt_f), cmap='seismic')
    plt.colorbar()
    plt.show()
    dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig (os.path.join (dir, 'f_test'))
    
 
    tol = config['data']['tol']
    save_path = '%s/data_%s_%s_%s_%s.mat' % (dir, N, tol['u0'], tol['ut0'], tol['f'])
    data = sio.loadmat(save_path)
    Coe_x, Coe_t = data['Coe_x'], data['Coe_t']

    index = []
    N_new = 0
    for i in range(N):
        coe_x, coe_t = Coe_x[:,i], Coe_t[:,:,i]
 
 
        u0_fun_train, ut0_fun_train, f_fun_train, u_fun_train = train(type_x, type_t, order_x, coe_x, coe_t)
        u0_train[i,:], ut0_train[i,:], f_train[i,:], u_train = u0_fun_train(x_u), ut0_fun_train(x_u), f_fun_train(x_f,t_f), u_fun_train(x_f,t_f)
 
        f_err = np.max(np.abs(f_test - f_train[i,:]))

        plt.figure()
 
        plt.pcolor(X_f, T_f, u_train.reshape(Nx_f, Nt_f), cmap='seismic')
        plt.colorbar()
        plt.show()
        dir = os.path.dirname(os.path.realpath(__file__))
        plt.savefig (os.path.join (dir, 'u_train'))

 
        if np.max(f_train[i,:]) >= 3 and np.min(f_train[i,:]) >= -0.2 and f_err <= 4:
            index.append(i)
            N_new += 1
 
 
    print(N_new)

    f_max = np.max(f_train, axis=1)
    f_min = np.min(f_train, axis=1)
    print('f_max_median:%s,f_min_median:%s'% (np.median(f_max), np.median(f_min)))
    plt.figure()
    plt.plot(f_max, label='f_max')
    plt.plot(f_min, label='f_min')
 
    plt.legend()
    plt.show()
    dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig (os.path.join (dir, 'two'))

    Coe_x_new, Coe_t_new = Coe_x[:,index], Coe_t[:,:,index]
    save_path = '%s/data_%s_%s_%s_%s.mat' % (dir, N_new, tol['u0'], tol['ut0'], tol['f'])
 
    sio.savemat(save_path, {'Coe_x':Coe_x_new,'Coe_t':Coe_t_new})

    u0_train, ut0_train, f_train = np.zeros((N_new,Nx_u)), np.zeros((N_new, Nx_u)), np.zeros((N_new,Nf))
    for i in range(N_new):
 
 
        coe_x, coe_t = Coe_x_new[:,i], Coe_t_new[:,:,i]
        u0_fun_train, ut0_fun_train, f_fun_train, u = train(type_x, type_t, order_x, coe_x, coe_t)

        u0_train[i,:], ut0_train[i,:], f_train[i,:] = u0_fun_train(x_u), ut0_fun_train(x_u), f_fun_train(x_f,t_f)
        print(np.max(f_train[i,:]))

    f_max = np.max(f_train, axis=1)
    f_min = np.min(f_train, axis=1)
    print('f_max_median:%s,f_min_median:%s'% (np.median(f_max), np.median(f_min)))
    plt.figure()
    plt.plot(f_max, label='f_max')
    plt.plot(f_min, label='f_min')
 
    plt.legend()
    plt.show()
    dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig (os.path.join (dir, 'two'))
    

if __name__ == "__main__":
    main()