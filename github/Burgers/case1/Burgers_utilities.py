
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
    xmin, xmax = domain['xmin']-2, domain['xmax']+2
    N = 100
    x_data = np.linspace(xmin, xmax, N)
    y_data = f(x_data)
    noise = np.random.normal(0, noise, N)
    y_data_noisy = y_data*(1+noise)
    f_noisy = interp1d(x_data, y_data_noisy, kind='cubic')

    return f_noisy


def add_noisy_2d(config, f, noise):
    # add smooth noisy for function f

 
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
 
 
    u0 = lambda x: np.zeros_like(x)
 
    f = lambda x, t: np.cos(np.pi*t)*np.exp(-x**2)

    return u0, f


def train_data(type_x, type_t, order_x, coe_x, coe_t):

    if type_x == 'hermite':
        ux = lambda x: hefunm(order_x-1, coe_x[0]*x+coe_x[1])  # [order_x, Nx]
        ux_x = lambda x: coe_x[0]*hefunm_diff(order_x-1, coe_x[0]*x+coe_x[1])
        ux_xx = lambda x: coe_x[0]**2*hefunm_diff2(order_x-1, coe_x[0]*x+coe_x[1])  # [order_x, Nx]

    if type_t == 'fourier':
        A, k1, k2 = coe_t[:,0].reshape(-1,1), coe_t[:,1].reshape(-1,1), coe_t[:,2].reshape(-1,1)
        ut = lambda t: A* (np.sin(k1*np.pi*t) + t* np.cos(k2*np.pi*t))
        ut_t = lambda t: A* (np.pi*k1*np.cos(np.pi*k1*t) + np.cos(k2*np.pi*t)-t*k2*np.pi*np.sin(k2*np.pi*t) )

    u = lambda x, t: np.sum(ux(x) * ut(t), axis = 0)

    u0 = partial(u, t=np.zeros(1))
    v = 0.2
    f = lambda x,t: np.sum(ux(x)*ut_t(t), axis = 0)+ u(x,t)*np.sum(ut(t)*ux_x(x), axis = 0)- v*np.sum(ut(t)*ux_xx(x), axis =0)

    return u0, f, u


def BurgersSolver(config, u0, f):
# def BurgersSolver(config, u0, f, u):
    # difference method used to verify the exact solution
    # u_t + u u_x - v u_xx = f

    v = 0.2

    domain = config['data']['domain']
    Nx, Nt = config['test']['Nx'], config['test']['Nt']

    Nx_ref = config['test']['Nx_ref']
    xmin_ref, xmax_ref = config['test']['domain_ref']['xmin'], config['test']['domain_ref']['xmax']

    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']
 
    dt = (tmax-tmin)/(Nt-1)
    dx = (xmax_ref-xmin_ref)/(Nx_ref-1)
    x_ref = np.linspace(xmin_ref, xmax_ref, Nx_ref)

 
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)

    Uh = np.zeros((Nt, Nx_ref))
    Uh[0,:] = u0(x_ref.T)
    
    A =  (np.diag(np.ones(Nx_ref-3), 1) - np.diag(np.ones(Nx_ref-3), -1))/2/dx
    B = (np.diag(np.ones(Nx_ref-3), 1)-2*np.diag(np.ones(Nx_ref-2), 0) + np.diag(np.ones(Nx_ref-3), -1))/dx**2
 
    x_f = x_ref[1:-1].flatten()
    for i in range(1, Nt):
        M = 1/dt*np.eye(Nx_ref-2) + np.diag(0.5*np.matmul(A, Uh[i-1,1:-1]),0) + 0.5*np.tile(Uh[i-1,1:-1].reshape(-1,1), (1, Nx_ref-2))*A - v*B/2
        F = (1/dt*Uh[i-1,1:-1] + v*np.matmul(B, Uh[i-1,1:-1])/2 + (f(x_f, t[i-1].item()) + f(x_f, t[i].item()))/2).flatten()
        Uh[i, 1:-1] = np.linalg.solve(M, F)
 


    U = np.zeros((Nt, Nx))
    for i in range (Nt):
        U[i,:] = np.interp(x, x_ref, Uh[i,:])

    return (x, t, U)


class DeepONetMulti(nn.Module):
    def __init__(self, branch1_layer, branch2_layer, trunk_layer, act):
        super(DeepONetMulti, self).__init__()
        self.branch1 = DenseNet(branch1_layer, act)
        self.branch2 = DenseNet(branch2_layer, act)
        self.trunk = DenseNet(trunk_layer, act)

    def forward(self, input1, input2, grid):
        a1 = self.branch1(input1)
        a2 = self.branch2(input2)
 
        b = self.trunk(grid)
 
 
 
        return torch.sum(a1 * a2* b, dim = 1)
 


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

    save_path = os.path.dirname(os.path.realpath(__file__))
    save_it = config['train']['save_it']
    save_error_it = config['train']['save_error_it']

    u_test, y_test, s_test = generate_one_test_data(config)
    u_test, y_test, s_test = torch.from_numpy(u_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(s_test).float()
    u_test, y_test, s_test = u_test.to(device), y_test.to(device), s_test.to(device)
    u1_test, u2_test = u_test[:, 0:m1], u_test[:, m1:m1+m2]
    error_it_log, error_L2_log, error_L1_log, error_max_log  = [], [], [], []
    
    for it in pbar:
 
 
        optimizer.zero_grad()
 

 
        u_bcs, y_bcs, s_bcs = next(train_loader_bcs)
        u_bcs, y_bcs, s_bcs = u_bcs.to(device), y_bcs.to(device), s_bcs.to(device)
 
        u1_bcs, u2_bcs = u_bcs[:,0:m1], u_bcs[:, m1:m1+m2]
        out_bcs = model(u1_bcs, u2_bcs, y_bcs)
 
        loss = torch.mean(torch.pow(torch.squeeze(s_bcs) - out_bcs, 2))
        
 
        loss.backward()
 
        optimizer.step()

        
        if (it+1) % 100 == 0:
            loss_it_log.append(it+1), loss_log.append(loss.item())
            pbar.set_postfix({'Data Loss': '{:.8f}'.format(loss)})

            if (it+1) % save_error_it == 0:
                s_pred = model(u1_test, u2_test, y_test)
                error_L2, error_L1, error_max = compute_error(s_test, s_pred)
                error_it_log.append(it+1)
                error_L2_log.append(error_L2.item()), error_L1_log.append(error_L1.item()), error_max_log.append(error_max.item()) 


        if (it+1) % save_it == 0:
 
            name = '%s_%s' % (config['train']['save_name'], it+1)
            save_checkpoint(save_path, name, model, optimizer)
 
 
            pltLoss(torch.tensor(loss_it_log, device = 'cpu'), torch.tensor(loss_log, device = 'cpu'), save_path, config['train']['save_name'])
            pltError(torch.tensor(error_it_log, device = 'cpu'),
                     torch.tensor(error_L2_log, device = 'cpu'),
                     torch.tensor(error_L1_log, device = 'cpu'),
                     torch.tensor(error_max_log, device = 'cpu'), save_path, config['train']['save_name'])


        scheduler.step()
    
    save_error_name = '%s/%sErrorData.mat' % (save_path, config['train']['save_name'])
    sio.savemat(save_error_name, {'error_it': error_it_log, 'error_L2': error_L2_log, 'error_L1': error_L1_log, 'error_max': error_max_log})
    save_loss_name = '%s/%sLossData.mat' % (save_path, config['train']['save_name'])
    sio.savemat(save_loss_name, {'loss_it': loss_it_log, 'loss': loss_log})


def mixed_train(model,              # model of neural operator
                train_loader_bcs,       # dataloader for training with data 
                train_loader_res,           # generator for  ICs
                optimizer,          # optimizer
                scheduler,          # learning rate scheduler
                config,             # configuration dict
                device,
                log=False,          # turn on the wandb
                project='PINO-default', # project name
                group='FDM',        # group name
                tags=['Nan'],       # tags
                use_tqdm=True):     # turn on tqdm
    

    weight_bcs = config['train']['weight_bcs']
    weight_res = config['train']['weight_res']
    m1, m2 = config['model']['m1'], config['model']['m2']
    model.train()
    pbar = range(config['train']['nIter'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    zero = torch.zeros(1).to(device)
    train_loader_bcs = sample_data(train_loader_bcs)
    train_loader_res = sample_data(train_loader_res)
    loss_bcs_log, loss_res_log, loss_total_log = [], [], []

    save_path = os.path.dirname(os.path.realpath(__file__))
    save_it = config['train']['save_it']
    for it in pbar:
 
 
        optimizer.zero_grad()
 

 
        u_bcs, y_bcs, s_bcs = next(train_loader_bcs)
        u_bcs, y_bcs, s_bcs = u_bcs.to(device), y_bcs.to(device), s_bcs.to(device)
 
        u1_bcs, u2_bcs = u_bcs[:,0:m1], u_bcs[:, m1:m1+m2]
        out_bcs = model(u1_bcs, u2_bcs, y_bcs)
 
        loss_bcs = torch.mean(torch.pow(torch.squeeze(s_bcs) - out_bcs, 2))

 
        u_res, y_res, s_res = next(train_loader_res)
        u_res, y_res, s_res = u_res.to(device), y_res.to(device), s_res.to(device)
        y_res.requires_grad = True

        u1_res, u2_res = u_res[:,0:m1], u_res[:,m1:]
        out_res = model(u1_res, u2_res, y_res)
        out_tx = torch.autograd.grad(outputs=out_res, inputs=y_res, grad_outputs = torch.ones_like(out_res), 
                            retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
        out_t = out_tx[:,1]
        out_x = out_tx[:,0]
        out_x_tx = torch.autograd.grad(outputs=out_x, inputs=y_res, grad_outputs = torch.ones_like(out_x), 
                            retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
        out_xx = out_x_tx[:,0]
        out_xx_tx = torch.autograd.grad(outputs=out_xx, inputs=y_res, grad_outputs = torch.ones_like(out_xx), 
                            retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
        out_xxx = out_xx_tx[:,0]
        
        res = out_t + 6*out_res*out_x + out_xxx
        loss_res = torch.mean(torch.pow(torch.squeeze(s_res) - res, 2))
       
 
        loss_total = weight_bcs* loss_bcs + weight_res*loss_res
        loss_total.backward()
 
        optimizer.step()

        if it % 100 == 0:
            loss_bcs_log.append(loss_bcs)
            loss_res_log.append(loss_res)
            loss_total_log.append(loss_total)
            pbar.set_postfix({'Data Loss': '{:.7f}'.format(loss_bcs),'Physics Loss': '{:.7f}'.format(loss_res),'Total Loss': '{:.7f}'.format(loss_total)})
        
        if (it+1) % save_it == 0:
            name = '%s_%s' % (config['train']['save_name'], it+1)
            save_checkpoint(save_path, name, model, optimizer)
     

        scheduler.step()
    
    pltLossPI(torch.tensor(loss_bcs_log, device = 'cpu'),  torch.tensor(loss_res_log, device = 'cpu'), save_path, config['train']['save_name'])


def pltLossPI(loss_bcs_log, loss_res_log, DataFolderPath, name):
 
 
    plt.figure(figsize = (6,5))
    plt.plot(loss_bcs_log, lw=2, label='bcs')
    plt.plot(loss_res_log, lw=2, label='res')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
 
    fig_name = '%sLoss' % name
 
    plt.savefig(os.path.join(DataFolderPath, fig_name))
    plt.show()



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


 
 
def generate_one_training_data(config, coe_x, coe_t):

 
    P_ic, P_i= config['train']['P_ic'], config['train']['P_i']
    domain = config['train']['domain']
    Nx, Nt = config['train']['Nx'], config['train']['Nt']
    m1, m2 = config['model']['m1'], config['model']['m2']
 
    order_x = config['data']['order_x']
    type_x, type_t = config['data']['type_x'], config['data']['type_t']

    u0, f_fn, U = train_data(type_x, type_t, order_x, coe_x, coe_t)

    # Sample points from the boundary conditions
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    x1 = np.linspace(xmin, xmax, m1)
    M2 = round(math.sqrt(m2))
    x_f = np.linspace(xmin, xmax, M2)
    t_f = np.linspace(tmin, tmax, M2)
    x_f, t_f = np.meshgrid(x_f, t_f)
    x_f, t_f = x_f.reshape(1,-1), t_f.reshape(1,-1)
    u = np.hstack((u0(x1), f_fn(x_f, t_f)))

    x_ic = np.random.uniform(xmin, xmax, (P_ic, 1))
    t_ic = np.zeros((P_ic, 1))
    y_ic_train = np.hstack([x_ic, t_ic])
    s_ic_train = u0(x_ic.T).reshape(P_ic,1)

    # Training data for interior
    if P_i == 0:
        # u_train = u.repeat(P_ic+2*P_bc, 1)
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



# Generate training data corresponding to N input samples
def generate_training_data(config):
    P = config['train']['P_ic']+config['train']['P_i']**2
    N = config['train']['N_input']
    m1, m2 = config['model']['m1'], config['model']['m2']
    u_bcs_train = np.zeros((N,P,m1+m2))
    y_bcs_train = np.zeros((N,P,2))
    s_bcs_train = np.zeros((N,P,1))

    data_num = N*P*(m1+m2+3)
    print(data_num)

    
 
    dir = os.path.dirname(os.path.realpath(__file__))
 
    tol = config['data']['tol']
    save_path = '%s/data_%s_%s_%s.mat' % (dir, N, tol['u0'], tol['f'])
    data = sio.loadmat(save_path)
    Coe_x, Coe_t = data['Coe_x'], data['Coe_t']


    for i in range(N):
 
 
        coe_x, coe_t = Coe_x[:,i], Coe_t[:,:,i]
 
        u_bcs_train[i,:,:], y_bcs_train[i,:,:], s_bcs_train[i,:,:] = generate_one_training_data(config, coe_x, coe_t)
 
 

    u_bcs_train = u_bcs_train.reshape(N * P, -1)
    y_bcs_train = y_bcs_train.reshape(N * P, -1)
    s_bcs_train = s_bcs_train.reshape(N * P, -1)


    return u_bcs_train, y_bcs_train, s_bcs_train



# Generate test data corresponding to one input sample
def generate_one_test_data(config):

    Nx, Nt = config['test']['Nx'], config['test']['Nt']
    domain = config['test']['domain']


    m1, m2 = config['model']['m1'], config['model']['m2']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    x1 = np.linspace(xmin, xmax, m1)
    M2 = round(math.sqrt(m2))
    x_f = np.linspace(xmin, xmax, M2)
    t_f = np.linspace(tmin, tmax, M2)


    u0, f = test_data()

    if config['test']['noise_f'] >= 1e-5:
        f = add_noisy_2d(config, f, config['test']['noise_f'])
        fh = f(x_f, t_f).flatten()
    else:
        x_f, t_f = np.meshgrid(x_f, t_f)
        x_f, t_f = x_f.reshape(1,-1), t_f.reshape(1, -1)
        fh = f(x_f, t_f).flatten()
    if config['test']['noise_u0'] >= 1e-5:
        u0 = add_noisy_1d(config, u0, config['test']['noise_u0'])

    x, t, U = BurgersSolver(config, u0, f)
    x, t = np.meshgrid(x, t)
    y_test = np.hstack([x.flatten()[:, None], t.flatten()[:, None]])
    s_test = U.flatten()

    u = np.hstack((u0(x1), fh))
 
    u_test = np.tile(u, (Nx*Nt, 1))

    return u_test, y_test, s_test



def generate_one_training_data_PI(config, coe_x, coe_t):

 
    P_ic, P_i= config['train']['P_ic'], config['train']['P_i']
    Q = config['train']['Q']
    domain = config['train']['domain']
    Nx, Nt = config['train']['Nx'], config['train']['Nt']
    m1, m2 = config['model']['m1'], config['model']['m2']

    order_x = config['data']['order_x']
    type_x, type_t = config['data']['type_x'], config['data']['type_t']

    u0, f_fn, U = train_data(type_x, type_t, order_x, coe_x, coe_t)

    # Sample points from the boundary conditions
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    x1 = np.linspace(xmin, xmax, m1)
    M2 = round(math.sqrt(m2))
    x_f = np.linspace(xmin, xmax, M2)
    t_f = np.linspace(tmin, tmax, M2)
    x_f, t_f = np.meshgrid(x_f, t_f)
    x_f, t_f = x_f.reshape(1,-1), t_f.reshape(1,-1)
    u = np.hstack((u0(x1), f_fn(x_f, t_f)))

    # Sample points from the initial conditions
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

    x_r = np.random.uniform(xmin, xmax, (Q, 1))
    t_r = np.random.uniform(tmin, tmax, (Q, 1))
    u_r_train = np.tile(u, (Q,1)) 
    y_r_train = np.hstack([x_r, t_r])  # sample points for computing PDE residual, can be different form
    s_r_train = f_fn(x_r.T, t_r.T).reshape(Q,1)

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train


# Generate training data corresponding to N input samples
def generate_training_data_PI(config):
    P = config['train']['P_ic']+config['train']['P_i']**2
    Q = config['train']['Q']
    N = config['train']['N_input']
    m1, m2 = config['model']['m1'], config['model']['m2']
    u_bcs_train = np.zeros((N,P,m1+m2))
    y_bcs_train = np.zeros((N,P,2))
    s_bcs_train = np.zeros((N,P,1))
    u_res_train = np.zeros((N,Q,m1+m2))
    y_res_train = np.zeros((N,Q,2))
    s_res_train = np.zeros((N,Q,1))

    data_num = N*P*(m1+m2+3)
    print(data_num)

    
    # load training data
    dir = os.path.dirname(os.path.realpath(__file__))
    tol = config['data']['tol']
    save_path = '%s/data_%s_%s_%s.mat' % (dir, N, tol['u0'], tol['f'])
    data = sio.loadmat(save_path)
    Coe_x, Coe_t = data['Coe_x'], data['Coe_t']


    for i in range(N):
        coe_x, coe_t = Coe_x[:,:,i], Coe_t[:,:,i]
        u_bcs_train[i,:,:], y_bcs_train[i,:,:], s_bcs_train[i,:,:],\
        u_res_train[i,:,:], y_res_train[i,:,:], s_res_train[i,:,:] = generate_one_training_data_PI(config, coe_x, coe_t)

    u_bcs_train = u_bcs_train.reshape(N * P, -1)
    y_bcs_train = y_bcs_train.reshape(N * P, -1)
    s_bcs_train = s_bcs_train.reshape(N * P, -1)

    u_res_train = u_res_train.reshape(N * Q, -1)
    y_res_train = y_res_train.reshape(N * Q, -1)
    s_res_train = s_res_train.reshape(N * Q, -1)


    return u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train

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


def hefunm_diff(n, x):
    # first order derivative of hermite function

    nn = np.arange(n+2).reshape(-1,1)
    Nx = np.size(x)

    y = -x*hefunm(0, x)
    if n == 0:
        return y
    
    polyn = hefunm(n+1, x)
    poly = np.tile(np.sqrt(nn[1:-1]/2), (1,Nx))* polyn[:-2,:] - np.tile(np.sqrt(nn[2:]/2), (1,Nx))* polyn[2:,:]

    return np.vstack((y, poly))


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
    fig, ax = plt.subplots(figsize = (6,5))
    plt.plot(loss_it, loss, lw=2)
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

def pltSolFT(XX, TT, S_test, S_pred, S_pred_FT, DataFolderPath, name):    
    fig = plt.figure(figsize=(18,4))
    plt.subplot(1,4,1)
    plt.pcolor(XX, TT, torch.real(S_test), cmap='Spectral')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Exact $s(x,t)$')
    plt.tight_layout()

    plt.subplot(1,4,2)
    plt.pcolor(XX,TT, torch.real(S_pred), cmap='Spectral')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predict $s(x,t)$ of PINO')
    plt.tight_layout()

    plt.subplot(1,4,3)
    plt.pcolor(XX,TT, torch.real(S_pred_FT), cmap='Spectral')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predict $s(x,t)$ of fine tuned model')
    plt.tight_layout()

    plt.subplot(1,4,4)
    plt.pcolor(XX, TT, np.abs(S_pred_FT - S_test), cmap='Spectral')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Absolute error of fine tuned solution')
    plt.tight_layout()
    plt.savefig (os.path.join (DataFolderPath, name))
    plt.show()

def count_params(model):
    tol_ = 0
    for p in model.parameters():
        tol_ += torch.numel(p)
    return tol_