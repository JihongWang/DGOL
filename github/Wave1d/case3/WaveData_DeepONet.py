
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
from scipy.interpolate import griddata
import time, argparse, os
from WaveData_utilities import *
import torch
import torch.nn as nn
import yaml, math
from argparse import ArgumentParser
from torch.utils.data import DataLoader, TensorDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 

    parser = argparse.ArgumentParser('Train or Test Arg! ')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
 
 
    dir = os.path.dirname(os.path.realpath(__file__))
    config_file = '%s/%s' % (dir, args.config_path)

    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)


    model = DeepONetMulti(config['model']['branch1_layers'], config['model']['branch2_layers'], config['model']['branch3_layers'], config['model']['trunk_layers'], config['model']['act']).to(device)
    print('Number of model parameters: %s' % (count_params(model)))

    P = config['train']['P_ic']+config['train']['P_i']**2
    N = config['train']['N_input']
    print('Number of data: %s' % (N*P))
    print('Number of epoch: %s' % (config['train']['batch_size']*config['train']['nIter']/(N*P)))
    ################################################################
    #  train
    ################################################################
    if not args.test: 

        start = time.perf_counter()
        
        u_bcs_train, y_bcs_train, s_bcs_train = generate_training_data(config)
 
 
        end = time.perf_counter()
        print('Training data has been generated! Running time: %s Seconds' % (end-start))

        start = time.perf_counter()
        u_bcs_train, y_bcs_train, s_bcs_train = torch.from_numpy(u_bcs_train).float(), torch.from_numpy(y_bcs_train).float(), torch.from_numpy(s_bcs_train).float()
 
        end = time.perf_counter()
        print(end-start)

 
        train_loader_bcs = DataLoader(TensorDataset(u_bcs_train, y_bcs_train, s_bcs_train), batch_size=config['train']['batch_size'], shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=config['train']['base_lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['step_size'], gamma=config['train']['scheduler_gamma'])

 
        print("Start training:")
        start = time.perf_counter()
        train(model,
                    train_loader_bcs,   # dataloader for training with data
                    optimizer,          # optimizer
                    scheduler,          # learning rate scheduler
                    config,           # configuration dict
                    device,
                    log=False,          # turn on the wandb
                    project='PINO-default', # project name
                    group='FDM',        # group name
                    tags=['Nan'],       # tags
                    use_tqdm=True)      # turn on tqdm
        end = time.perf_counter()
        print('Training completed! Running time: %s Seconds' % (end-start))


    else:
        # Test
        # load trained model
        ckpt_path = '%s/checkpoints/%s_%s.pt' % (dir, config['train']['save_name'], config['test']['it'])
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])

        # Predict for one sample
        u_test, y_test, s_test = generate_one_test_data(config)
        u_test, y_test = torch.from_numpy(u_test).float(), torch.from_numpy(y_test).float()
        u_test, y_test = u_test.to(device), y_test.to(device)
        m1, m2 = config['model']['m1'], config['model']['m2']
        u1_test, u2_test, u3_test = u_test[:, 0:m1], u_test[:, m1:m1+m2], u_test[:, m1+m2:]
        s_pred = model(u1_test, u2_test, u3_test, y_test)
        # Plot on CPU
        name = '%sSol' % (config['train']['save_name'])
        pltSol(config, y_test.cpu(), s_pred.cpu(), s_test, dir, name)
        # Compute relative l2 error
        s_pred = s_pred.cpu().detach().numpy()
        error_L2 = np.linalg.norm(np.abs(s_test - s_pred)) / np.linalg.norm(np.abs(s_test))
        error_L1 = np.linalg.norm(s_test - s_pred, ord=1) / np.linalg.norm(s_test, ord=1)
        error_max = np.max(np.abs(s_test - s_pred))
        print('relative L2 error: {:.2e}'.format(error_L2))
        print('relative L1 error: {:.2e}'.format(error_L1))
        print('max error: {:.2e}'.format(error_max))

        # compute equation residual
        u_test, y_test, s_test = generate_one_test_PI_data(config)
        u_test, y_test = torch.from_numpy(u_test).float(), torch.from_numpy(y_test).float()
        u_test, y_test = u_test.to(device), y_test.to(device)
        m1, m2 = config['model']['m1'], config['model']['m2']
        u1_test, u2_test, u3_test = u_test[:, 0:m1], u_test[:, m1:m1+m2], u_test[:, m1+m2:]
        s_test = torch.from_numpy(s_test).float().to(device)
        eq_res = eq_residual(model, u1_test, u2_test, u3_test, y_test, s_test)
        eq_res = eq_res.cpu().detach().numpy()
        print('L2 error of Eq. residual: {:.2e}'.format(np.linalg.norm(eq_res)))
        print('MSE of Eq. residual: {:.2e}'.format(np.mean(eq_res**2)))
        print('max error of Eq. residual: {:.2e}'.format(np.max(np.abs(eq_res))))


if __name__ == "__main__":
    main()