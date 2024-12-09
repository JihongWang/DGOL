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
 
    args = parser.parse_args()


    u0, ut0, f, u = test_data()

    x = np.linspace(-30, 30, 201)
    t = 1
    U = u(x,t).flatten()
    U0 = u0(x).flatten()

    plt.figure(figsize=(6,5))
    plt.rcParams.update({'font.size': 15}) 
    plt.plot(x, U, label='$u(x, t=1)$')
    plt.plot(x, U0, label='$u(x, t=0)$')
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig (os.path.join (dir, 'u'))

    

if __name__ == "__main__":
    main()