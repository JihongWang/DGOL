
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.interpolate import griddata
from scipy import interpolate
 

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

try:
    import wandb
except ImportError:
    wandb = None


 
 


def test_data():
 
    k1, k2 = np.pi, np.pi
    u0 = lambda x1, x2: np.exp(-(x1**2+x2**2)/2)
    ut0 = lambda x1, x2: np.exp(-(x1**2+x2**2)/2)* (x1+x2)
 
    f = lambda x1, x2, t: np.exp(-((x1-t)**2+(x2-t)**2)/2)* ((x1-t+x2-t)**2*np.cos(k1*t)-2*k1*(x1-t+x2-t)*np.sin(k1*t) - (2+k1**2)*np.cos(k1*t) -np.cos(k1*t)*(-2+(x1-t)**2+(x2-t)**2))

    u = lambda x1, x2, t: np.exp(-((x1-t)**2+(x2-t)**2)/2)*np.cos(k1*t)

    return u0, ut0, f, u


def train_data(coe):
    A, a1, a2, s, k1, k2, w = coe[0], coe[1], coe[2], coe[3], coe[4], coe[5], coe[6]
    u = lambda x, y, t:  A*np.exp(-((x-a1*t)**2+(y-a2*t)**2)/s**2)*np.cos(k1*x+k2*y-w*t)

    ut = lambda x, y, t:  A*( ((2*a1*(x - a1*t) + 2*a2*(y - a2*t)) * np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.cos(-k1*x - k2*y + t*w) / s**2)\
                             -(w * np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.sin(-k1*x - k2*y + t*w)) )

    utt = lambda x, y, t:  A*( (((-2*a1**2 - 2*a2**2) * np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.cos(-k1*x - k2*y + t*w)) / s**2)\
                              +(w**2 * (-np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2)) * np.cos(-k1*x - k2*y + t*w))\
                                 -((2*w*(2*a1*(x - a1*t) + 2*a2*(y - a2*t)) * np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.sin(-k1*x - k2*y + t*w)) / s**2)\
                                     +(((2*a1*(x - a1*t) + 2*a2*(y - a2*t))**2 * np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.cos(-k1*x - k2*y + t*w)) / s**4) )

    uxx = lambda x, y, t: A*( (k1**2 * (-np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2)) * np.cos(-k1*x - k2*y + t*w)) \
                             -( (4*k1*(x - a1*t)*np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.sin(-k1*x - k2*y + t*w)) / s**2)\
                                -( (2*np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.cos(-k1*x - k2*y + t*w)) / s**2 )\
                                    +((4*(x - a1*t)**2 * np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.cos(-k1*x - k2*y + t*w)) / s**4 ))

    uyy = lambda x, y, t: A*( (k2**2 * (-np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2)) * np.cos(-k1*x - k2*y + t*w))\
                             -((4*k2*(y - a2*t)*np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.sin(-k1*x - k2*y + t*w)) / s**2)\
                                 -((2*np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.cos(-k1*x - k2*y + t*w)) / s**2)\
                                     + ((4*(y - a2*t)**2 * np.exp(-((x - a1*t)**2 + (y - a2*t)**2)/s**2) * np.cos(-k1*x - k2*y + t*w)) / s**4) )


    u0 = partial(u, t = np.zeros(1))
    ut0 = partial(ut, t = np.zeros(1))
    f = lambda x, y, t: utt(x,y,t)-uxx(x,y,t)-uyy(x,y,t)

    return u0, ut0, f, u


def WaveSolver(config, u0, ut0, f, plot):
    # difference method used to verify the exact solution
    # u_tt - u_xx = f

    domain = config['test']['domain']
    Nx, Nt = config['test']['Nx'], config['test']['Nt']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    Nx_ref = 201
    xmin_ref, xmax_ref = -10, 10
    dt = (tmax-tmin)/(Nt-1)
    dx = (xmax_ref-xmin_ref)/(Nx_ref-1)
    x_ref = np.linspace(xmin_ref, xmax_ref, Nx_ref)
    X, Y = np.meshgrid(x_ref, x_ref)
    X, Y = X.reshape(1,-1), Y.reshape(1,-1)


    Uh = np.zeros((Nt, Nx_ref**2))
    Uh[0,:] = u0(X,Y).reshape(1,-1)
    Uh[1,:] = Uh[0,:] + dt*ut0(X,Y).reshape(1,-1)
 
    
    a0 = np.ones(Nx_ref)
    a1 = np.ones(Nx_ref-1)
    A0 = -4*np.diag(a0,0)+np.diag(a1,1)+np.diag(a1,-1)
 
    A = np.kron(np.eye(Nx_ref),A0)+ np.kron(np.diag(a1,1),np.eye(Nx_ref))+ np.kron(np.diag(a1,-1),np.eye(Nx_ref))
    c = 1
 
    for n in range(2, Nt):
 
        Uh[n,:] = 2*Uh[n-1,:]-Uh[n-2,:] + dt**2*(c**2/dx**2*np.matmul(A, Uh[n-1,:]) + f(X,Y,(n-1)*dt))


 
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    U = np.zeros((Nt, Nx*Nx))
    for i in range (Nt):
        f_interp = interpolate.interp2d(x_ref, x_ref, Uh[i,:].reshape(Nx_ref, Nx_ref))
        U[i,:] = f_interp(x, x).reshape(1,-1)
    

    if plot == 1:
        X, Y = np.meshgrid(x, x)

        plt.pcolor(X, Y, U[-1,:].reshape(Nx, Nx), cmap='seismic')
        plt.title('U')
        plt.colorbar()
        plt.show()
        dir = os.path.dirname(os.path.realpath(__file__))
        plt.savefig (os.path.join (dir, 'u_test'))

    return (x, t, U)


def VerifySol(config, u0, ut0, f, u):

    x, t, uh = WaveSolver(config, u0, ut0, f, plot=0)

    domain = config['test']['domain']
    Nx, Nt = config['test']['Nx'], config['test']['Nt']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    X, Y = np.meshgrid(x, x)
    x, y = X.reshape(1,-1), Y.reshape(1,-1)
    U_exact = u(x, y, tmax)
    
    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    Uh_plot = uh[-1,:].reshape(Nx, Nx)
    plt.pcolor(X, Y, Uh_plot, cmap='seismic')
    plt.title('Uh')
    plt.colorbar()

    plt.subplot(1,3,2)
    U_plot = U_exact.reshape(Nx, Nx)
    plt.pcolor(X, Y, U_plot, cmap='seismic')
    plt.colorbar()
    plt.title('U_exact')

    plt.subplot(1,3,3)
    error = np.abs(U_plot - Uh_plot)
    plt.pcolor(X, Y, error, cmap='seismic')
    plt.title('error')
    plt.colorbar()
    plt.show()
    dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig (os.path.join (dir, 'u_test'))
    print(np.max(error))
    print('---')



class DeepONetMulti(nn.Module):
    def __init__(self, branch1_layer, branch2_layer, trunk_layer, act):
        super(DeepONetMulti, self).__init__()
        self.branch1 = DenseNet(branch1_layer, act)
        self.branch2 = DenseNet(branch1_layer, act)
        self.branch3 = DenseNet(branch2_layer, act)
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

# The key to making data cycle
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

# loader_bcs, loader_res trained together
def mixed_train_together(model,              # model of neural operator
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


    u_test, y_test, s_test = generate_one_test_data(config)
    print('Test data generated!')
    u_test, y_test, s_test = torch.from_numpy(u_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(s_test).float()
    u_test, y_test, s_test = u_test.to(device), y_test.to(device), s_test.to(device)
    u1_test, u2_test, u3_test = u_test[:, 0:m1], u_test[:, m1:2*m1], u_test[:, 2*m1:]
    error_it_log, error_L2_log, error_L1_log, error_max_log  = [], [], [], []
    save_path = os.path.dirname(os.path.realpath(__file__))
    save_it, save_error_it = config['train']['save_it'], config['train']['save_error_it']

    loss_it_log, loss_log = [], []
    
    for it in pbar:
 
 
        optimizer.zero_grad()
 

 
        u_bcs, y_bcs, s_bcs = next(train_loader_bcs)
        u_bcs, y_bcs, s_bcs = u_bcs.to(device), y_bcs.to(device), s_bcs.to(device)
 
        u1_bcs, u2_bcs, u3_bcs = u_bcs[:,0:m1], u_bcs[:, m1:2*m1], u_bcs[:, 2*m1:] 
        out_bcs = model(u1_bcs, u2_bcs, u3_bcs, y_bcs)
 
        loss = torch.mean(torch.pow(torch.squeeze(s_bcs) - out_bcs, 2))
        
 
        loss.backward()
 
        optimizer.step()

        if (it+1) % 100 == 0:
            loss_it_log.append(it+1), loss_log.append(loss.item())
            pbar.set_postfix({'Data Loss': '{:.8f}'.format(loss)})
            if (it+1) % save_error_it == 0:
                s_pred = model(u1_test, u2_test, u3_test, y_test)
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


def compute_error(s_test, s_pred):
    error_L2 = torch.norm(s_test - s_pred, p=2) / torch.norm(s_test, p=2)
    error_L1 = torch.norm(s_test - s_pred, p=1) / torch.norm(s_test, p=1)
    error_max = torch.max(torch.abs(s_test - s_pred))

    return error_L2, error_L1, error_max


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
 
    m1, m2 = config['model']['m1'], config['model']['m2']
    

    order_x = config['data']['order_x']
    type_x, type_t = config['data']['type_x'], config['data']['type_t']

    u0, ut0, f_fn, U = train_data(coe)

 
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    M1 = round(math.sqrt(m1))
    x1 = np.linspace(xmin, xmax, M1)
    X1, X2 = np.meshgrid(x1, x1)
    x1, x2 = X1.reshape(1,-1), X2.reshape(1,-1)
    M2 = round(np.cbrt(m2))
    x_f = np.linspace(xmin, xmax, M2)
    t_f = np.linspace(tmin, tmax, M2)
    x1_f, x2_f, t_f = np.meshgrid(x_f, x_f, t_f)
    x1_f, x2_f, t_f = x1_f.reshape(1,-1), x2_f.reshape(1,-1), t_f.reshape(1,-1)
    u = np.hstack((u0(x1, x2), ut0(x1, x2), f_fn(x1_f, x2_f, t_f)))

    # Sample points from the initial conditions
    x1_ic = np.random.uniform(xmin, xmax, (P_ic, 1))
    x2_ic = np.random.uniform(xmin, xmax, (P_ic, 1))
    x1_ic, x2_ic = np.meshgrid(x1_ic, x2_ic)
    x1_ic, x2_ic = x1_ic.reshape(1,-1), x2_ic.reshape(1,-1)
    t_ic = np.zeros((P_ic**2, 1))
    y_ic_train = np.hstack([x1_ic.T, x2_ic.T, t_ic])
    s_ic_train = u0(x1_ic, x2_ic).reshape(P_ic**2,1)

    # Training data for interior
    if P_i == 0:
        u_train = np.tile(u, (P_ic**2, 1))
        y_train = y_ic_train
        s_train = s_ic_train
    else:
        x_i = np.linspace(xmin, xmax, P_i)
        t_i = np.linspace(tmin, tmax, P_i)
        x1_i, x2_i, t_i = np.meshgrid(x_i, x_i, t_i)
        x1_i, x2_i, t_i = x1_i.reshape(1,-1), x2_i.reshape(1,-1), t_i.reshape(1,-1)
        s_i_train = U(x1_i, x2_i, t_i).reshape(-1,1)
        y_i_train = np.hstack([x1_i.reshape(-1,1), x2_i.reshape(-1,1), t_i.reshape(-1,1)])

        P = P_ic**2 + P_i**3
        u_train = np.tile(u, (P, 1))
        y_train = np.vstack([y_ic_train, y_i_train])
        s_train = np.vstack([s_ic_train, s_i_train])

    return u_train, y_train, s_train



# Generate training data corresponding to N input samples
def generate_training_data(config):
    P = config['train']['P_ic']**2+config['train']['P_i']**3
    N = config['train']['N_input']
    m1, m2 = config['model']['m1'], config['model']['m2']
    u_bcs_train = np.zeros((N,P,2*m1+m2))
    y_bcs_train = np.zeros((N,P,3))
    s_bcs_train = np.zeros((N,P,1))

    data_num = N*P*(2*m1+m2+3)
    print(data_num)

    
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

    m1, m2 = config['model']['m1'], config['model']['m2']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    u0, ut0, f, U = test_data()

    M1 = round(math.sqrt(m1))
    x1 = np.linspace(xmin, xmax, M1)
    x2 = x1
    X1, X2 = np.meshgrid(x1, x2)
    x1, x2 = X1.reshape(1,-1), X2.reshape(1,-1)

    M2 = round(np.cbrt(m2))
    x_f = np.linspace(xmin, xmax, M2)
    t_f = np.linspace(tmin, tmax, M2)
    x1_f, x2_f, t_f = np.meshgrid(x_f, x_f, t_f)
    x1_f, x2_f, t_f = x1_f.reshape(1,-1), x2_f.reshape(1,-1), t_f.reshape(1, -1)

    u = np.hstack((u0(x1, x2).flatten(), ut0(x1, x2).flatten(), f(x1_f, x2_f, t_f).flatten()))

    u_test = np.tile(u, (Nx**2*Nt, 1))

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    x1, x2, t = np.meshgrid(x, x, t)
    x1, x2, t = x1.reshape(1,-1), x2.reshape(1,-1), t.reshape(1,-1)
    s_test = U(x1, x2, t).flatten()
    y_test = np.hstack([x1.reshape(-1,1), x2.reshape(-1,1), t.reshape(-1,1)])

    return u_test, y_test, s_test


def generate_plot_data(config):

    Nx = config['test']['Nx_plot']
    domain = config['test']['domain']

    m1, m2 = config['model']['m1'], config['model']['m2']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']

    # u known, input dimension for u0,ut0,f is fixed, y can be changed.
    u0, ut0, f, U = test_data()

    M1 = round(math.sqrt(m1))
    x1 = np.linspace(xmin, xmax, M1)
    x2 = x1
    X1, X2 = np.meshgrid(x1, x2)
    x1, x2 = X1.reshape(1,-1), X2.reshape(1,-1)

    M2 = round(np.cbrt(m2))
    x_f = np.linspace(xmin, xmax, M2)
    t_f = np.linspace(tmin, tmax, M2)
    x1_f, x2_f, t_f = np.meshgrid(x_f, x_f, t_f)
    x1_f, x2_f, t_f = x1_f.reshape(1,-1), x2_f.reshape(1,-1), t_f.reshape(1, -1)

    u = np.hstack((u0(x1, x2).flatten(), ut0(x1, x2).flatten(), f(x1_f, x2_f, t_f).flatten()))

    u_test = np.tile(u, (Nx**2, 1))

    x = np.linspace(xmin, xmax, Nx)
    t = tmax* np.ones((1, Nx**2))
    x1, x2 = np.meshgrid(x, x)
    x1, x2 = x1.reshape(1,-1), x2.reshape(1,-1)
    s_test = U(x1, x2, t).flatten()
    y_test = np.hstack([x1.reshape(-1,1), x2.reshape(-1,1), t.reshape(-1,1)])

    return u_test, y_test, s_test


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


def pltSol(config, y_test, s_pred, s_test, DataFolderPath, name): 
    domain = config['test']['domain']
    xmin, xmax, tmin, tmax = domain['xmin'], domain['xmax'], domain['tmin'], domain['tmax']
 
    Nx = config['test']['Nx_plot']
    x = np.linspace(xmin, xmax, Nx)
 
    XX, YY = np.meshgrid(x, x)
 
    s_pred, s_test = s_pred.reshape(Nx, Nx), s_test.reshape(Nx, Nx)
 
 
 

    fig = plt.figure(figsize=(18,5))
    ax = plt.subplot(1,3,1)
    im = plt.pcolor(XX,YY, np.real(s_test), cmap='Spectral')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('$x_1$', fontsize=15)
    ax.set_ylabel('$X_2$', fontsize=15)
    ax.set_title(f"Exact $u(x_1,x_2,t={tmax})$", fontsize=15)
    ax.tick_params(labelsize=15) 
    plt.tight_layout()

    ax = plt.subplot(1,3,2)
    im = plt.pcolor(XX,YY, np.real(s_pred), cmap='Spectral')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('$x_1$', fontsize=15)
    ax.set_ylabel('$X_2$', fontsize=15)
    ax.set_title(f'Predict $u(x_1,x_2,t={tmax})$', fontsize=15)
    ax.tick_params(labelsize=15) 
    plt.tight_layout()

    ax = plt.subplot(1,3,3)
    im = plt.pcolor(XX,YY, np.abs(s_pred - s_test), cmap='Spectral')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('$x_1$', fontsize=15)
    ax.set_ylabel('$X_2$', fontsize=15)
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