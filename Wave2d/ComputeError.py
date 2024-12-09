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

    save_error_name = '%s/%sErrorData.mat' % (dir, config['train']['save_name'])
    data = sio.loadmat(save_error_name)
    error_it, error_L2, error_L1, error_max = data['error_it'].flatten(), data['error_L2'].flatten(), data['error_L1'].flatten(), data['error_max'].flatten()

 
    N = 20
    average_error_L2, average_error_L1, average_error_max = np.mean(error_L2[-N:]), np.mean(error_L1[-N:]), np.mean(error_max[-N:])

    print('average_error_L2: %s' % average_error_L2 )
    print('average_error_L1: %s' % average_error_L1 )
    print('average_error_max: %s' % average_error_max )


if __name__ == "__main__":
    main()