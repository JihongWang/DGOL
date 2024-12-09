import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.interpolate import griddata

import time, argparse

import os
from numpy.polynomial.legendre import Legendre

import torch
import torch.nn as nn
import yaml, math
from argparse import ArgumentParser
import numpy as np
from timeit import default_timer
import torch.nn.functional as F
from itertools import cycle
from scipy.integrate import odeint
import scipy.io as sio

from scipy.interpolate import interp1d, interp2d

try:
    import wandb
except ImportError:
    wandb = None

def add_noisy_1d(config, f, noise):
    domain = config['test']['domain_ref']
    xmin, xmax = domain['xmin']-5, domain['xmax']+5
    N = 100
    x_data = np.linspace(-10, 10, N)
    y_data = f(x_data)
    noise = np.random.normal(0, noise, N)
    y_data_noisy = y_data*(1+noise)
    f_noisy = interp1d(x_data, y_data_noisy, kind='cubic')

    return f_noisy


def add_noisy_2d(config, f, noise):
    domain = config['test']['domain_ref']
    xmin, xmax = domain['xmin'], domain['xmax']
    tmin, tmax = domain['tmin'], domain['tmax']
    Nx = 100
    x_data = np.linspace(xmin, xmax, Nx)
    Nt = 10
    t_data = np.linspace(tmin, tmax, Nt)
    [X, T] = np.meshgrid(x_data,t_data)
    y_data = f(X.flatten(), T.flatten()).reshape(Nt, Nx)
    noise = np.random.normal(0, noise, y_data.shape)
    y_data_noisy = y_data*(1+ noise)
    f_noisy = interp2d(x_data, t_data, y_data_noisy, kind='cubic')

    return f_noisy


def test_data(): 
    u0 = lambda x: np.sin(x)
    u0_x = lambda x: np.cos(x)
    ut0 = lambda x: np.cos(x)
    ut0_x = lambda x: -np.sin(x)

    f = lambda x,t: np.exp(-x**2)
 

    return u0, ut0, f, u0_x, ut0_x


def train_data(coe):

    A, k1, w1, b1, k2, w2, b2 = coe[0], coe[1], coe[2], coe[3], coe[4], coe[5], coe[6]
    A, k1, w1, b1, k2, w2, b2  = A.reshape(-1,1), k1.reshape(-1,1), w1.reshape(-1,1), b1.reshape(-1,1), k2.reshape(-1,1), w2.reshape(-1,1), b2.reshape(-1,1)
    u = lambda x, t: A* np.sin(k1*x+w1*t+b1)+np.cos(k2*x+w2*t+b2)
    u_t = lambda x, t: A* np.cos(k1*x+w1*t+b1)*w1-np.sin(k2*x+w2*t+b2)*w2
    u_tt = lambda x, t: -A* np.sin(k1*x+w1*t+b1)*w1**2-np.cos(k2*x+w2*t+b2)*w2**2
    u_xx = lambda x, t: -A* np.sin(k1*x+w1*t+b1)*k1**2-np.cos(k2*x+w2*t+b2)*k2**2

    u0 = partial(u, t=np.zeros(1))
    ut0 = partial(u_t, t=np.zeros(1))
    f = lambda x, t: u_tt(x,t) - u_xx(x,t)
    return u0, ut0, f, u


def train_data_0(coe):

    A, a, k, c1, c2, w, b = coe[0], coe[1], coe[2], coe[3], coe[4], coe[5], coe[6]
 
    A, a, k, c1, c2, w, b = A.reshape(-1,1), a.reshape(-1,1), k.reshape(-1,1), c1.reshape(-1,1), c2.reshape(-1,1), w.reshape(-1,1), b.reshape(-1,1)
    a = -np.abs(a)
 
    u = lambda x, t: np.sum(A*np.exp(a*(x+c1*t+c2)**2)*np.cos(k*x-w*t+b), axis=0)
    u_t = lambda x, t: np.sum(A*np.exp(a*(c1*t + c2 + x)**2)*(2*a*c1*(c1*t + c2 + x)*np.cos(b + k*x - t*w) + w*np.sin(b + k*x - t*w)), axis=0)
    u_tt = lambda x, t: np.sum( A *np.exp(a*(c1*t + c2 + x)**2)*(2*a*c1**2*np.cos(b + k*x - t*w) + 2*a*c1*w*(c1*t + c2 + x)*np.sin(b + k*x - t*w) + 2*a*c1*(c1*t + c2 + x)*(2*a*c1*(c1*t + c2 + x)*np.cos(b + k*x - t*w) + w*np.sin(b + k*x - t*w)) - w**2*np.cos(b + k*x - t*w)) , axis=0)
    u_xx = lambda x, t: np.sum( A*np.exp(a*(c1*t + c2 + x)**2)*(-2*a*k*(c1*t + c2 + x)*np.sin(b + k*x - t*w) + 2*a*(c1*t + c2 + x)*(2*a*(c1*t + c2 + x)*np.cos(b + k*x - t*w) - k*np.sin(b + k*x - t*w)) + 2*a*np.cos(b + k*x - t*w) + k**2*(-np.cos(b + k*x - t*w))) , axis=0)

    u0 = partial(u, t=np.zeros(1))
    ut0 = partial(u_t, t=np.zeros(1))
 
    f = lambda x, t: u_tt(x,t) - u_xx(x,t)

    u0_x = lambda x: np.sum( A* np.exp(a*(x+c2)**2)*(2*a*(x+c2)*np.cos(k*x+b)-k*np.sin(k*x+b)), axis=0)

 
    return u0, ut0, f, u



def train_data_s(type_x, type_t, order_x, coe_x, coe_t):

    a = 1

    if type_x == 'hermite':
        ux = lambda x: hefunm(order_x-1, coe_x[0]*x+coe_x[1])  # [1, Nx]
        ux_xx = lambda x: coe_x[0]**2*hefunm_diff2(order_x-1, coe_x[0]*x+coe_x[1])  # [1, Nx]

    if type_x == 'exp':
        std = coe_x.reshape(-1,1)
 
 
        ux = lambda x: np.exp(-(x)**2/std**2)
        ux_xx = lambda x: -2/std**2*(1-2*(x)**2/std**2)*np.exp(-(x)**2/std**2)

    if type_x == 'fourier':
        k1, k2, b = coe_x[0], coe_x[1], coe_x[2]
        k1, k2, b = k1.reshape(-1,1), k2.reshape(-1,1), b.reshape(-1,1)
        ux = lambda x: (np.sin(k1*x+b)+np.cos(k2*x))
        ux_xx = lambda x: -(k1**2*np.sin(k1*x+b)+k2**2*np.cos(k2*x))
        ux_x = lambda x: k1*np.cos(k1*x+b)-k2*np.sin(k2*x)

    if type_x == 'exp_sin':
        mean, std, k = coe_x[0], coe_x[1], coe_x[2]
        mean, std, k = mean.reshape(-1,1), std.reshape(-1,1), k.reshape(-1,1)
        ux = lambda x: np.exp(-(x-mean)**2/std**2)*np.sin(k*x)
        ux_xx = lambda x: np.exp(-(x-mean)**2/std**2)*(np.sin(k*x)*(4*mean**2-8*mean*x-std**4*k**2-2*std**2+4*x**2)-4*std**2*k*(x-mean)*np.cos(k*x))/std**4


    if type_t == 'other':
        A, coe = coe_t[:,0].reshape(-1,1), coe_t[:,1].reshape(-1,1)
        ut = lambda t: A*t* np.sin(coe*t)
        ut_tt = lambda t: A*(2*coe*np.cos(coe*t)-coe**2*t*np.sin(coe*t))
        ut0 = lambda x: np.zeros_like(x).flatten()

    if type_t == 'exp1':
 
        A, coe = coe_t[:,0].reshape(-1,1), coe_t[:,1].reshape(-1,1)
        ut = lambda t: A*t**2* np.exp(coe*t)
        ut_tt = lambda t: A*np.exp(coe*t)*(coe**2*t**2+4*coe*t+2)
        ut0 = lambda x: np.zeros_like(x).flatten()

    if type_t == 'exp':
 
 
 
        A, mean, std = coe_t[:,0], coe_t[:,1], coe_t[:,2] 
        A, mean, std = A.reshape(-1,1), mean.reshape(-1,1), std.reshape(-1,1)
        ut = lambda t: A*t**2*np.exp(-(t-mean)**2/std**2)
        ut_tt = lambda t: 2*A*np.exp(-(t-mean)**2/std**2) *(1-2*t*(t-mean)/std**2-(3*t**2-2*mean*t)/std**2+2*t**2*(t-mean)**2/std**4)
 
        ut0 = lambda x: np.zeros_like(x).flatten()

    if type_t == 'exp_sin':
 
 
 
        A, mean, std, k = coe_t[:,0], coe_t[:,1], coe_t[:,2], coe_t[:,3]
        A, mean, std, k = A.reshape(-1,1), mean.reshape(-1,1), std.reshape(-1,1), k.reshape(-1,1)
        ut = lambda t: A*t**2*np.exp(-(t-mean)**2/std**2)*np.sin(k*t)
        ut_tt = lambda t: A*np.exp(-(t-mean)**2/std**2)*(-2*(t-mean)/std**2*(-t**2*np.sin(k*t)*2*(t-mean)/std**2+2*t*np.sin(k*t)+k*t**2*np.cos(k*t))\
                                                         -2/std**2*(3*t**2-2*mean*t)*np.sin(k*t)-2*k/std**2*(t**3-mean*t**2)*np.cos(k*t)+2*np.sin(k*t)+4*t*k*np.cos(k*t)-k**2*t**2*np.sin(k*t))
 
        ut0 = lambda x: np.zeros_like(x).flatten()

    if type_t == 'exp1_sin':
 
 
 
        A, mean, std, k = coe_t[:,0], coe_t[:,1], coe_t[:,2], coe_t[:,3]
        A, mean, std, k = A.reshape(-1,1), mean.reshape(-1,1), std.reshape(-1,1), k.reshape(-1,1)
        ut = lambda t: A*t*np.exp(-(t-mean)**2/std**2)*np.sin(k*t)
        ut_tt = lambda t: A*np.exp(-(t-mean)**2/std**2)*(-2*(t-mean)/std**2*(k*t*np.cos(k*t)+np.sin(k*t)-2*(t-mean)*t/std**2*np.sin(k*t))\
                                                         -k**2*t*np.sin(k*t)+2*k*np.cos(k*t)-2/std**2*((2*t-mean)*np.sin(k*t)+(t**2-mean*t)*k*np.cos(k*t)))
 
        ut0 = lambda x: np.zeros_like(x).flatten()
    
    if type_t == 'fourier':
        A, k1, k2 = coe_t[0], coe_t[1], coe_t[2]
        A, k1, k2 = A.reshape(-1,1), k1.reshape(-1,1), k2.reshape(-1,1)
        ut = lambda t: A* (np.sin(k1*t)+np.cos(k2*t))
        ut_tt = lambda t: -A*(k1**2*np.sin(k1*t)+k2**2*np.cos(k2*t))
        ut0 = lambda x: np.matmul((A*k1).T, ux(x)).flatten()
    
    if type_t == 'poly':
        A = coe_t.reshape(-1,1)
        ut = lambda t: A* t**2
        ut_tt = lambda t: 2*A*np.ones_like(t)
        ut0 = lambda x: np.zeros_like(x).flatten()

    if type_t == 'poly_cos':
        A, a, b = coe_t[:,0].reshape(-1,1), coe_t[:,1].reshape(-1,1), coe_t[:,2].reshape(-1,1)
        ut = lambda t: A* t**2 *np.cos(a*t+b)
        ut_tt = lambda t: A* ((2-a**2*t**2)*np.cos(a*t+b) -4*a*t*np.sin(a*t+b))
        ut0 = lambda x: np.zeros_like(x).flatten()

    u = lambda x, t: np.sum(ux(x) * ut(t), axis = 0)

    u0 = partial(u, t=np.zeros(1))
    f = lambda x,t: np.sum(ux(x)*ut_tt(t), axis = 0)- a**2*np.sum(ut(t)*ux_xx(x), axis =0)

    u0_x = lambda x: ux_x(x)*ut(0)

 
    return u0, ut0, f, u


def WaveSolverABC_train(config, u0, ut0, f, u, u0_x):
    # initial functions are not decay, f is compactly supported in computational domain
    #  solver: analytical ABC + FDM 

    a = 1
    domain = config['train']['domain']
    Nt = config['train']['Nt']

    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']
    dt = (tmax-tmin)/(Nt-1)
    xmin, xmax = -10, 10
    Nx = 2001
    h = (xmax - xmin) / (Nx-1)
    x = np.linspace(xmin, xmax, Nx)
    th = np.linspace(tmin, tmax, Nt)
    U = np.zeros((Nt, Nx))

    U[0, :] = u0(x)
    U[1, :] = dt * ut0(x) + U[0, :]

    for n in range(1, Nt-1):
        U[n + 1, 1:-1] = 2 * U[n, 1:-1] - U[n - 1, 1:-1] + (a * dt / h)**2 * (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2]) + dt**2 * f(x[1:-1], dt * n)
        U[n + 1, -1] = 1 / (a * dt + h) * (a * dt * U[n + 1, -2] + h * U[n, -1] + a * dt * h * (u0_x(xmax + a * dt * n) + 1 / a * ut0(xmax + a * dt * n)))
        U[n + 1, 0] = 1 / (a * dt + h) * (a * dt * U[n + 1, 1] + h * U[n, 0] - a * dt * h * (u0_x(xmin - a * dt * n) - 1 / a * ut0(xmin - a * dt * n)))

    Nx = config['train']['Nx']
    xmin, xmax = domain['xmin'], domain['xmax']
    xh = np.linspace(xmin, xmax, Nx)
    Uh = np.zeros((Nt, Nx))
    for i in range (Nt):
        Uh[i,:] = np.interp(xh, x, U[i,:])


    return xh, th, Uh


def WaveSolverABC(config, u0, ut0, f, u0_x):
    # initial functions are not decay, f is compactly supported in computational domain
    #  solver: analytical ABC + FDM 

    a = 1
    domain = config['test']['domain_ref']

    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']
    Nt = config['test']['Nt_ref']
    dt = (tmax-tmin)/(Nt-1)
    Nx = config['test']['Nx_ref']
    h = (xmax - xmin) / (Nx-1)
    x = np.linspace(xmin, xmax, Nx)
    U = np.zeros((Nt, Nx))

    U[0, :] = u0(x)
    U[1, :] = dt * ut0(x) + U[0, :]

    for n in range(1, Nt-1):
        U[n + 1, 1:-1] = 2 * U[n, 1:-1] - U[n - 1, 1:-1] + (a * dt / h)**2 * (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2]) + dt**2 * f(x[1:-1], dt * n)
        U[n + 1, -1] = 1 / (a * dt + h) * (a * dt * U[n + 1, -2] + h * U[n, -1] + a * dt * h * (u0_x(xmax + a * dt * n) + 1 / a * ut0(xmax + a * dt * n)))
        U[n + 1, 0] = 1 / (a * dt + h) * (a * dt * U[n + 1, 1] + h * U[n, 0] - a * dt * h * (u0_x(xmin - a * dt * n) - 1 / a * ut0(xmin - a * dt * n)))


    Nt = config['test']['Nt']
    th = np.linspace(tmin, tmax, Nt)
    Nx = config['test']['Nx']
    xmin, xmax = config['test']['domain']['xmin'], config['test']['domain']['xmax']
    xh = np.linspace(xmin, xmax, Nx)
    Uh = np.zeros((Nt, Nx))
    for i in range (Nt):
        Uh[i,:] = np.interp(xh, x, U[2*i,:])

    return xh, th, Uh



def WaveSolver(config, u0, ut0, f, u):
    # difference method used to verify the exact solution
    # u_tt - u_xx = f


    domain = config['test']['domain']
    Nx, Nt = config['test']['Nx'], config['test']['Nt']

    Nx_ref = 4001
    xmin_ref, xmax_ref = -10, 10

    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']
    dt = (tmax-tmin)/(Nt-1)
    dx = (xmax_ref-xmin_ref)/(Nx_ref-1)
    x_ref = np.linspace(xmin_ref, xmax_ref, Nx_ref)

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)

    Uh = np.zeros((Nt, Nx_ref))
    Uh[0,:] = u0(x_ref.T)
    Uh[1,:] = Uh[0,:] + dt*ut0(x_ref.T)
    
    A = -2*np.diag(np.ones(Nx_ref-2)) + np.diag(np.ones(Nx_ref-3), 1) + np.diag(np.ones(Nx_ref-3), -1)
    x_f = x_ref[1:-1].reshape(1, Nx_ref-2)
    for i in range(2, Nt):
        Uh[i, 1:-1] = dt**2/dx**2*np.matmul(A, Uh[i-1,1:-1])+ 2*Uh[i-1,1:-1] - Uh[i-2,1:-1] + dt**2*f(x_f, t[i-1]*np.ones((1, Nx_ref-2)))


    U = np.zeros((Nt, Nx))
    for i in range (Nt):
        U[i,:] = np.interp(x, x_ref, Uh[i,:])
   return (x, t, U)


def WaveSolver_train(config, u0, ut0, f, u):
    # difference method used to verify the exact solution
    # u_tt - u_xx = f

    domain = config['train']['domain']
    Nx, Nt = config['test']['Nx'], config['test']['Nt']

    Nx_ref = 4001
    xmin_ref, xmax_ref = -10, 10

    Nt = 401

    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']
    dt = (tmax-tmin)/(Nt-1)
    dx = (xmax_ref-xmin_ref)/(Nx_ref-1)
    x_ref = np.linspace(xmin_ref, xmax_ref, Nx_ref)

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)

    Uh = np.zeros((Nt, Nx_ref))
    Uh[0,:] = u0(x_ref.T)
    Uh[1,:] = Uh[0,:] + dt*ut0(x_ref.T)
    
    A = -2*np.diag(np.ones(Nx_ref-2)) + np.diag(np.ones(Nx_ref-3), 1) + np.diag(np.ones(Nx_ref-3), -1)
    x_f = x_ref[1:-1].reshape(1, Nx_ref-2)
    for i in range(2, Nt):
        Uh[i, 1:-1] = dt**2/dx**2*np.matmul(A, Uh[i-1,1:-1])+ 2*Uh[i-1,1:-1] - Uh[i-2,1:-1] + dt**2*f(x_f, t[i-1]*np.ones((1, Nx_ref-2)))


    U = np.zeros((Nt, Nx))
    for i in range (Nt):
        U[i,:] = np.interp(x, x_ref, Uh[i,:])

    U_exact = u(x,tmax)
    plt.figure()
    plt.plot(x, U[-1,:], label='U')
    plt.plot(x, U_exact, label='U_exact')
    plt.legend()
    plt.show()
    dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig (os.path.join (dir, 'u_train'))

    print('---')

    return (x, t, U)



def ReferenceSol(u0, ut0, domain, Nx, Nt):
    '''
    Reference solution generated by pseudo-spectral method on large enough domain
    u_tt = u_xx
    '''
    xmin, xmax, tmin, tmax = domain
    L_ref = 10
    N_ref = 1000
    x_ref = np.linspace (-L_ref, L_ref, N_ref+1)
    x_ref = x_ref[0:-1]
    t = np.linspace(tmin, tmax, Nt)

    u0_fft = np.fft.fft(np.squeeze(np.array(list(map(u0, x_ref)))))
    U0_fft = np.tile(u0_fft, (Nt, 1))
    ut0_fft = np.fft.fft (ut0 (x_ref))
    Ut0_fft = np.tile (ut0_fft, (Nt, 1))

    N2 = N_ref // 2
    k = np.pi / L_ref * np.concatenate((np.linspace(0, N2, N2), np.linspace(-N2, -1, N2)))
    KK, TT = np.meshgrid(k, t)
    u_ref = np.real(np.fft.ifft(U0_fft*np.cos(KK*TT)+ Ut0_fft * np.sinc(KK * TT/np.pi)* TT))

    # interpolate to x
    x = np.linspace (xmin, xmax, Nx)
    UU = np.zeros ((Nt, Nx))
    for i in range (Nt):
 
        UU = UU.at[i, :].set (np.interp (x, x_ref, u_ref[i, :]))

    return x, t, UU


class DeepONetMulti(nn.Module):
    def __init__(self, branch1_layer, branch2_layer, branch3_layer, trunk_layer, act):
        super(DeepONetMulti, self).__init__()
        self.branch1 = DenseNet(branch1_layer, act)
        self.branch2 = DenseNet(branch2_layer, act)
        self.branch3 = DenseNet(branch3_layer, act)
        self.trunk = DenseNet(trunk_layer, act)

    def forward(self, input1, input2, input3, grid):
        a1 = self.branch1(input1)
        a2 = self.branch2(input2)
        a3 = self.branch3(input3)
 
        b = self.trunk(grid)
 
 
 
        return torch.sum(a1 * a2* a3* b, dim = 1)
 


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        if isinstance(nonlinearity, str):
            if nonlinearity == 'relu':
                nonlinearity = nn.ReLU
            elif nonlinearity == 'tanh':
                nonlinearity = nn.Tanh
            elif nonlinearity == 'sigmoid':
                nonlinearity = nn.Sigmoid
            else:
                raise ValueError(f'{nonlinearity} is not supported')
        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x

 
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

 
def train(model,              # model of neural operator
                train_loader_bcs,       # dataloader for training with data
                optimizer,          # optimizer
                scheduler,          # learning rate scheduler
                config,             # configuration dict
                device,
                log=False,          # turn on the wandb
                project='PINO-default', # project name
                group='FDM',        # group name
                tags=['Nan'],       # tags
                use_tqdm=True):     # turn on tqdm

    m1, m2 = config['model']['m1'], config['model']['m2']
    model.train()
    pbar = range(config['train']['nIter'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    zero = torch.zeros(1).to(device)
    train_loader_bcs = sample_data(train_loader_bcs)
 
   
    
    loss_it_log, loss_log = [], []
    error_it_log, error_L2_log, error_L1_log, error_max_log  = [], [], [], []

    save_path = os.path.dirname(os.path.realpath(__file__))
    save_it = config['train']['save_it']
    save_error_it = config['train']['save_error_it']


    # used for test in training
    u_test, y_test, s_test = generate_one_test_data(config)
    u_test, y_test, s_test = torch.from_numpy(u_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(s_test).float()
    u_test, y_test, s_test = u_test.to(device), y_test.to(device), s_test.to(device)
    u1_test, u2_test, u3_test = u_test[:, 0:m1], u_test[:, m1:m1+m2], u_test[:, m1+m2:]

    
    for it in pbar:
 
 
        optimizer.zero_grad()
 

 
        u_bcs, y_bcs, s_bcs = next(train_loader_bcs)
        u_bcs, y_bcs, s_bcs = u_bcs.to(device), y_bcs.to(device), s_bcs.to(device)
 
        u1_bcs, u2_bcs, u3_bcs = u_bcs[:,0:m1], u_bcs[:, m1:m1+m2], u_bcs[:, m1+m2:] 
        out_bcs = model(u1_bcs, u2_bcs, u3_bcs, y_bcs)
 
        loss = torch.mean(torch.pow(torch.squeeze(s_bcs) - out_bcs, 2))
        
 
        loss.backward()
 
        optimizer.step()

        if (it+1) % 100 == 0:
            loss_it_log.append((it+1)), loss_log.append(loss.item())
            pbar.set_postfix({'Data Loss': '{:.8f}'.format(loss)})
            if (it+1) % save_error_it == 0:
                s_pred = model(u1_test, u2_test, u3_test, y_test)
                error_L2, error_L1, error_max = compute_error(s_test, s_pred)
                error_it_log.append(it+1)
                error_L2_log.append(error_L2.item()), error_L1_log.append(error_L1.item()), error_max_log.append(error_max.item()) 

             
        if (it+1) % save_it == 0:
 
            name = '%s_%s' % (config['train']['save_name'], it+1)
            save_checkpoint(save_path, name, model, optimizer)
 
 
            pltLoss(torch.tensor(loss_it_log, device = 'cpu'),torch.tensor(loss_log, device = 'cpu'), save_path, config['train']['save_name'])
            pltError(torch.tensor(error_it_log, device = 'cpu'),
                     torch.tensor(error_L2_log, device = 'cpu'),
                    torch.tensor(error_L1_log, device = 'cpu'),
                    torch.tensor(error_max_log, device = 'cpu'), save_path, config['train']['save_name'])


        scheduler.step()
    
    save_error_name = '%s/%sErrorData.mat' % (save_path, config['train']['save_name'])
    sio.savemat(save_error_name, {'error_it': error_it_log, 'error_L2': error_L2_log, 'error_L1': error_L1_log, 'error_max': error_max_log})
    save_loss_name = '%s/%sLossData.mat' % (save_path, config['train']['save_name'])
    sio.savemat(save_loss_name, {'loss_it': loss_it_log, 'loss': loss_log})



def save_checkpoint(save_path, name, model, optimizer=None):
    ckpt_dir = '%s/checkpoints/'% (save_path)
    print(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0
    name = '%s.pt' % name
    torch.save({
        'model': model_state_dict,
        'optim': optim_dict
    }, ckpt_dir + name)
    print('Checkpoint is saved at %s' % ckpt_dir + name)


 
 
def generate_one_training_data(config, coe):

 
    P_ic, P_i= config['train']['P_ic'], config['train']['P_i']
    domain = config['train']['domain']
    Nx, Nt = config['train']['Nx'], config['train']['Nt']
    m1, m2, m3 = config['model']['m1'], config['model']['m2'], config['model']['m3']
    
 
 

    order_x = config['data']['order_x']
    type_x, type_t = config['data']['type_x'], config['data']['type_t']

 
    u0, ut0, f_fn, U = train_data(coe)

 

 
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    x1 = np.linspace(xmin, xmax, m1)
    x2 = np.linspace(xmin, xmax, m2)
    M3 = round(math.sqrt(m3))
    x_f = np.linspace(xmin, xmax, M3)
    t_f = np.linspace(tmin, tmax, M3)
    x_f, t_f = np.meshgrid(x_f, t_f)
    x_f, t_f = x_f.reshape(1,-1), t_f.reshape(1,-1)
    u = np.hstack((u0(x1), ut0(x2), f_fn(x_f, t_f)))
 
    x_ic = np.random.uniform(xmin, xmax, (P_ic, 1))
    t_ic = np.zeros((P_ic, 1))
    y_ic_train = np.hstack([x_ic, t_ic])
    s_ic_train = u0(x_ic.T).reshape(P_ic,1)

    # Training data for interior
    if P_i == 0:
 
        u_train = np.tile(u, (P_ic, 1))
        y_train = y_ic_train
        s_train = s_ic_train
    else:
        
        x_i = np.linspace(xmin, xmax, P_i)
        t_i = np.linspace(tmin, tmax, P_i)
        x_i, t_i = np.meshgrid(x_i, t_i)
        x_i, t_i = x_i.reshape(1,-1), t_i.reshape(1,-1)
        s_i_train = U(x_i, t_i).reshape(-1,1)
        y_i_train = np.hstack([x_i.reshape(-1,1), t_i.reshape(-1,1)])

        P = P_ic + P_i**2
 
        u_train = np.tile(u, (P, 1))
        y_train = np.vstack([y_ic_train, y_i_train])
        s_train = np.vstack([s_ic_train, s_i_train])

    return u_train, y_train, s_train



 
def generate_training_data(config):
 
    P = config['train']['P_ic']+config['train']['P_i']**2
    N = config['train']['N_input']
    m1, m2, m3 = config['model']['m1'], config['model']['m2'], config['model']['m3']
 
    u_bcs_train = np.zeros((N,P,m1+m2+m3))
    y_bcs_train = np.zeros((N,P,2))
    s_bcs_train = np.zeros((N,P,1))

    
 
    dir = os.path.dirname(os.path.realpath(__file__))
 
    tol = config['data']['tol']
    save_path = '%s/data_%s_%s_%s_%s.mat' % (dir, N, tol['u0'], tol['ut0'], tol['f'])
    data = sio.loadmat(save_path)
 
    Coe = data['Coe']


    for i in range(N):
 
        coe = Coe[:,i]
 
        u_bcs_train[i,:,:], y_bcs_train[i,:,:], s_bcs_train[i,:,:] = generate_one_training_data(config, coe)
 
 

    u_bcs_train = u_bcs_train.reshape(N * P, -1)
    y_bcs_train = y_bcs_train.reshape(N * P, -1)
    s_bcs_train = s_bcs_train.reshape(N * P, -1)


    return u_bcs_train, y_bcs_train, s_bcs_train



# Generate test data corresponding to one input sample
def generate_one_test_data(config):

    Nx, Nt = config['test']['Nx'], config['test']['Nt']
    domain = config['test']['domain']


    m1, m2, m3 = config['model']['m1'], config['model']['m2'], config['model']['m3']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    x1 = np.linspace(xmin, xmax, m1)
    x2 = np.linspace(xmin, xmax, m2)

    M3 = round(math.sqrt(m3))
    x_f = np.linspace(xmin, xmax, M3)
    t_f = np.linspace(tmin, tmax, M3)

 
 

    u0, ut0, f, u0_x, _ = test_data()
    if config['test']['noise_f'] >= 1e-5:
        f = add_noisy_2d(config, f, config['test']['noise_f'])
        fh = f(x_f, t_f).flatten()
    else:
        x_f, t_f = np.meshgrid(x_f, t_f)
        x_f, t_f = x_f.reshape(1,-1), t_f.reshape(1, -1)
        fh = f(x_f, t_f).flatten()

    if config['test']['noise_u0'] >= 1e-5:
        u0 = add_noisy_1d(config, u0, config['test']['noise_u0'])
    if config['test']['noise_ut0'] >= 1e-5:
        ut0 = add_noisy_1d(config, ut0, config['test']['noise_ut0'])

    x, t, U = WaveSolverABC(config,u0,ut0,f,u0_x)

    u = np.hstack((u0(x1), ut0(x2), fh))
    u_test = np.tile(u, (Nx*Nt, 1))


    XX, TT = np.meshgrid(x, t)
    y_test = np.hstack([XX.flatten()[:, None], TT.flatten()[:, None]])
    s_test = U.flatten()

    return u_test, y_test, s_test



# Generate test data corresponding to one input sample
def generate_test_data(config):

    Nx, Nt = config['test']['Nx'], config['test']['Nt']
    domain = config['test']['domain']
    N_test = config['test']['N_test']

    m1, m2, m3 = config['model']['m1'], config['model']['m2'], config['model']['m3']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    x1 = np.linspace(xmin, xmax, m1)
    x2 = np.linspace(xmin, xmax, m2)
    M3 = round(math.sqrt(m3))
    x_f = np.linspace(xmin, xmax, M3)
    t_f = np.linspace(tmin, tmax, M3)
 
 

    u_test = np.zeros((N_test, Nx*Nt, (m1+m2+m3)))
    y_test = np.zeros((N_test, Nx*Nt, 2))
    s_test = np.zeros((N_test, Nx*Nt, 1))

    for i in range(N_test):
 
 
        u0, ut0, f, u0_x, _ = test_data()
        f = add_noisy_2d(config, f)
        x, t, U = WaveSolverABC(config,u0,ut0,f,u0_x)
        s_test[i,:,:] = U.reshape(-1,1)

        u = np.hstack((u0(x1), ut0(x2), f(x_f, t_f).flatten()))
        u_test[i,:,:] = np.tile(u, (Nx*Nt, 1))

        XX, TT = np.meshgrid(x, t)
        y_test[i,:,:] = np.hstack([XX.flatten()[:, None], TT.flatten()[:, None]])

 

    u_test = u_test.reshape(N_test*Nx*Nt, -1)
    y_test = y_test.reshape(N_test*Nx*Nt, 2)
    s_test = s_test.reshape(N_test*Nx*Nt, 1)

    return u_test, y_test, s_test



def compute_error(s_test, s_pred):
    error_L2 = torch.norm(s_test - s_pred, p=2) / torch.norm(s_test, p=2)
    error_L1 = torch.norm(s_test - s_pred, p=1) / torch.norm(s_test, p=1)
    error_max = torch.max(torch.abs(s_test - s_pred))

    return error_L2, error_L1, error_max



# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    X1, X2 = np.meshgrid(np.squeeze(x1), np.squeeze(x2))
 
 
    r2 = ((X1-X2)/lengthscales)**2
 
    return output_scale * np.exp(-0.5 * r2)


def hefunm(n, x):
    '''
    hermite function on [-infty, infty]
    :param n: degree from 0 to n
    :param x: row vector
    :return: [n+1,length(x)]
    '''

    cst = 1 / np.sqrt (np.sqrt (np.pi))
 
 
 
 

    if n == 0:
        y = cst * np.exp(-x**2 / 2)
        return y
    if n == 1:
        y = np.vstack((cst * np.exp(-x**2 / 2), cst * np.sqrt(2) * x* np.exp(-x**2 / 2)))
        return y

    polylst = cst * np.exp (-x**2 / 2)
    poly = cst * np.sqrt (2) * x* np.exp (-x**2 / 2)
    y = np.vstack((polylst, poly))

    for k in range(1,n):
        polyn = np.sqrt (2 / (k + 1)) * x * poly - np.sqrt (k / (k + 1)) * polylst
        polylst = poly
        poly = polyn
        y = np.vstack((y, poly))

    return y


def hefunm_diff2(n, x):
    # second order derivative of hermite function
    nn = np.arange(n+1)
    X, NN = np.meshgrid(x, nn)
    y = -np.multiply((2*NN +1- X**2), hefunm(n, x))  # (7.74)

    return y


def poly(n, x):
    # polynomial basis
    # return: [n+1, length(x)]

    if n == 0:
        y = np.ones_like(x)
        return y
    y = np.ones_like(x)
    for k in range(1,n+1):
        y = np.vstack((y, np.power(x, k)))

    return y


def lepoly(n, x, domain):
    # legendre function on domain
    # return: [n+1, length(x)]

    xl, xr = domain
    x = 2/(xr-xl)*x+(xl+xr)/(xl-xr)

    if n == 0:
        y = np.ones_like(x)
        return y
    if n==1:
        y = np.vstack((np.ones_like(x), x))
        return y
    polylst = np.ones_like(x)
    poly = x
    y = np.vstack((polylst, poly))
    for k in range(1,n):
        polyn = ((2*(k+1)-1)*x*poly-k*polylst)/(k+1)
        polylst=poly
        poly=polyn
        y = np.vstack((y, poly))

    return y




def mkdir(fn):
    if not os.path.isdir(fn):
        os.mkdir(fn)


def save_var(var_name, var, DataFolderPath):
    text_file = open ("%s/%s.txt" % (DataFolderPath, var_name), "w")
    for para in var:
        if np.size (var[para]) > 100:
            continue
        text_file.write ('%s: %s\n' % (para, var[para]))
    text_file.close () 


def save_var_test(var_name, var, DataFolderPath):
    text_file = open ("%s/%s.txt" % (DataFolderPath, var_name), "w")
    for para in var:
 
 
        text_file.write ('%s: %s\n' % (para, var[para]) + '\n')
    text_file.close()


def load_var(file_name):
 
    with open(file_name, 'r') as f:
        lines = f.readlines()

    my_dict = {}

 
    for line in lines:
 
        items = line.split(': ')

 
        key = items[0]
        values = [float(item) for item in items[1:]]

 
        my_dict[key] = values

    return my_dict

def setPath(subFolderName=3, DataFolderName=230115):
    dir = os.path.dirname(os.path.realpath(__file__))
    subFolderPath = '%s/%s' % (dir, subFolderName)
    mkdir(subFolderPath)
    DataFolderPath = '%s/%s' % (subFolderPath, DataFolderName)
    mkdir(DataFolderPath) 
    modelFoldername = 'model'
    modelFolderPath = '%s/%s' % (DataFolderPath, modelFoldername)
    mkdir(modelFolderPath)
    return DataFolderPath, modelFolderPath


def pltVar(var_train, var_test, DataFolderPath, name):
 
 
    plt.figure(figsize = (6,5))
    c1, c2, c3, c4, c5 = 'cornflowerblue', 'paleturquoise', 'mediumslateblue', 'orangered', 'orange'
    yy = np.ones(len(var_test['u0_max']))
    plt.plot(var_train['u0_bcs_max']* yy, '--', color = c1, lw=2, label='train_u0_max')
    plt.plot(var_train['f_bcs_max']* yy, '--', color = c2, lw=2, label='train_f_max')
    plt.plot(var_train['s_bcs_max']* yy, '--', color = c3, lw=2, label='train_s_max')

    plt.plot(var_test['u0_max'], marker='.', color = c1, lw=2, label='u0_max')
    plt.plot(var_test['f_max'], marker='.', color = c2, lw=2, label='f_max')
    plt.plot(var_test['s_max'], marker='.', color = c3, lw=2, label='s_max')
    plt.plot(var_test['error_L2'], marker='.', color = c4, lw=2, label='error_L2')
    plt.plot(var_test['error_max'], marker='.', color = c5, lw=2, label='error_max')

    plt.xlabel('N')
 
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
 
    fig_name = '%svar' % name
    plt.savefig(os.path.join(DataFolderPath, fig_name))
    plt.show()

def format_func(value, tick_number):
 
    return f'{int(value/1000)}k'

def pltError(error_it_log, error_L2_log, error_L1_log, error_max_log, DataFolderPath, name):
 
    plt.rcParams.update({'font.size': 15}) 
    fig, ax = plt.subplots(figsize = (6,5))
    plt.plot(error_it_log, error_L2_log, '-x', lw=2, label='Relative $L^2$ error')
    plt.plot(error_it_log, error_L1_log, '-o', lw=2, label='Relative $L^1$ error')
    plt.plot(error_it_log, error_max_log, '-+', lw=2, label='Max error')
    plt.xlabel('No. of iterations')
    plt.ylabel('Error')
    
 
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
 
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
 
    fig_name = '%sError' % name
 
    plt.savefig(os.path.join(DataFolderPath, fig_name))
    plt.show()



def pltLoss(loss_it, loss, DataFolderPath, name):
 
 

    plt.rcParams.update({'font.size': 15})
    plt.rcParams['font.family'] = 'Roman'

    fig, ax = plt.subplots(figsize = (6,5))
    plt.plot(loss_it, loss, lw=2, color='steelblue')
    plt.xlabel('No. of iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
 
 
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
 
    plt.tight_layout()
 
    fig_name = '%sLoss' % name
 
    plt.savefig(os.path.join(DataFolderPath, fig_name))
    plt.show()



def pltSol(config, y_test, s_pred, s_test, DataFolderPath, name): 
    domain = config['test']['domain']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']
 
    x = torch.linspace(xmin, xmax, config['test']['Nx'])
    t = torch.linspace (tmin, tmax, config['test']['Nt'])
    TT, XX = torch.meshgrid(t, x)
    y_test, s_pred = y_test.detach(), s_pred.detach()
 
    S_pred = griddata(y_test, s_pred.flatten(), (XX,TT), method='cubic')
    S_test = griddata(y_test, s_test.flatten(), (XX,TT), method='cubic')

    fig = plt.figure(figsize=(18,5))
    ax = plt.subplot(1,3,1)
    im = plt.pcolor(XX, TT, np.real(S_test), cmap='Spectral')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$t$', fontsize=15)
    ax.set_title('Exact $u(x,t)$', fontsize=15)
    ax.tick_params(labelsize=15)    
    plt.tight_layout()

    ax = plt.subplot(1,3,2)
    im = plt.pcolor(XX,TT, np.real(S_pred), cmap='Spectral')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$t$', fontsize=15)
    ax.set_title('Predict $u(x,t)$', fontsize=15)
    ax.tick_params(labelsize=15) 
    plt.tight_layout()

    ax = plt.subplot(1,3,3)
    im = plt.pcolor(XX, TT, np.abs(S_pred - S_test), cmap='Spectral')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$t$', fontsize=15)
    ax.set_title('Absolute error', fontsize=15)
    ax.tick_params(labelsize=15) 
    plt.tight_layout()
    plt.savefig (os.path.join (DataFolderPath, name))
    plt.show()

def count_params(model):
    tol_ = 0
    for p in model.parameters():
        tol_ += torch.numel(p)
    return tol_