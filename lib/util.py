import os
import logging
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, **configs):
    if configs['train']['adjust_lr'] == 'type1':
        lr_adjust = {epoch: configs['train']['lr'] * (0.5 ** ((epoch - 1) // 1))}
    elif configs['train']['adjust_lr'] == 'type2':
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                     10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif configs['train']['adjust_lr'] == 'type3':
        lr_adjust = {epoch: configs['train']['lr'] if epoch < 3 else configs['train']['lr']*(0.9 ** ((epoch-3)//1))}
    elif configs['train']['adjust_lr'] == '3':
        lr_adjust = {epoch: configs['train']['lr'] if epoch < 10 else configs['train']['lr']*0.1}
    elif configs['train']['adjust_lr'] == '4':
        lr_adjust = {epoch: configs['train']['lr'] if epoch < 15 else configs['train']['lr']*0.1}
    elif configs['train']['adjust_lr'] == '5':
        lr_adjust = {epoch: configs['train']['lr'] if epoch < 25 else configs['train']['lr']*0.1}
    elif configs['train']['adjust_lr'] == '6':
        lr_adjust = {epoch: configs['train']['lr'] if epoch < 5 else configs['train']['lr']*0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Epoch: {epoch} Updating learning rate to {lr}')

def adjust_learning_rate_patchtst(optimizer, scheduler, epoch):
    lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Epoch: {epoch} Updating learning rate to {lr}')


def test_params_flop(model, x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print(f'INFO: Trainable parameter count: {model_params / 1000000.0:.2f}M')
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.epo = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        self.epo += 1
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + f'checkpoint_epo{self.epo}.pth')
        torch.save(model.state_dict(), path + '/' + f'best_checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(y_label, y_pred=None, fig_name='../pic/test.pdf'):
    plt.figure()
    plt.plot(y_label, label='GroundTruth', linewidth=2)
    if y_pred is not None:
        plt.plot(y_pred, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(fig_name, bbox_inches='tight')

#
def get_logger_simple(file_dir, bsl_name):
    logger = logging.getLogger(bsl_name)
    logger.setLevel(logging.INFO)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
        print(f'Establish Log, Log_dir:{file_dir}')
    file_handler = logging.FileHandler(os.path.join(file_dir, f'{bsl_name}.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s  - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# def get_logger_complex(kwargs):
#     # initial the log dir
#     type = kwargs['data']['type']
#     in_lens = kwargs['model']['in_lens']
#     out_lens = kwargs['model']['out_lens']
#     graph_conv_type = kwargs['data']['graph_conv_type']
#     run_id = f'{type}_{in_lens}in_{out_lens}out_{graph_conv_type}conv'
#     base_dir = kwargs['log_dir']
#     log_dir = os.path.join(base_dir, run_id)
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#         print(f'Establish Log, Log_dir:{log_dir}')
#     logger = logging.getLogger('Graph WaveNet')
#     logger.setLevel(logging.INFO)
#     file_handler = logging.FileHandler(os.path.join(log_dir, 'Graph WaveNet.log'))
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_formatter = logging.Formatter('%(asctime)s  - %(levelname)s - %(message)s')
#     console_handler.setFormatter(console_formatter)
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
#
#     return logger
