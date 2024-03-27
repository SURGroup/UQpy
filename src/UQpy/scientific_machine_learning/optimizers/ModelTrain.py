import sys
sys.path.append("../")
from utils.utils_process import construct_paths, write_config, load_data, load_test_data, get_filepath, construct_det_path, get_largest_params, get_largest_conv_params,  comp_tau_list
from HMC_torch import hamiltorch
from HMC_torch.hamiltorch import util
from architectures.u_net import BayesByBackprop, Deterministic, MonteCarloDropout
from architectures.u_net.layers import metrics
from enum import Enum
import os
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
import numpy as np

class BaseTrainer:
    def __init__(self, net, device, cfg):
        self.net = net.to(device)
        self.device = device
        self.cfg = cfg
        self.ckpt_name = self.load_checkpoint()

    def load_checkpoint(self):
        filepath = get_filepath(self.cfg.method)
        ckpt_name, _, _ = construct_paths(self.cfg)
        write_config(self.cfg, filepath)
        if os.path.exists(ckpt_name) and self.cfg.preload:
            self.net.load_state_dict(torch.load(ckpt_name, map_location=self.device))
        return ckpt_name
    
    def run_training(self):
        _, train_loader, val_loader, out_channels, _, X_tr_tensor, Y_tr_tensor, _, _, _, _, _, _, _, _ = load_data(self.cfg)
        optimizer = Adam(self.net.parameters(), lr=self.cfg.lr_start)
        lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.patience, verbose=True)
        best_valid_loss = float('inf')
        
        for epoch in range(self.cfg.n_epochs):
            print(f'Epoch {epoch}')
            train_loss, train_acc = self.train_model(train_loader, optimizer, mode='train')
            valid_loss, valid_acc = self.train_model(val_loader, optimizer, mode='validate')
            lr_sched.step(valid_loss)
            print(valid_loss)
            # Checkpoint saving logic
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.net.state_dict(), self.ckpt_name)
                print(f'Model saved at {self.ckpt_name}')

            print(f'Completed Epoch {epoch}')

    def train_model(self, data_loader, optimizer, mode='train'):
        return
        
class DeterministicTrainer(BaseTrainer):
    def __init__(self, net, device, cfg):
        super().__init__(net, device, cfg)
        self.criterion = nn.MSELoss().to(device)
        
    def train_model(self, data_loader, optimizer, mode='train'):
        if mode not in ['train', 'validate']:
            raise ValueError("Mode must be 'train' or 'validate'")
        self.net.train() if mode == 'train' else self.net.eval()
        total_loss = 0.0
        accs = []
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)            
            if mode == 'train':
                optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, target)
            if mode == 'train':
                loss.backward()
                optimizer.step()
                            
            total_loss += loss.item() * data.size(0)
            accs.append(metrics.acc(output.detach(), target))
        return total_loss, np.mean(accs)
        
class MCDTrainer(DeterministicTrainer):
    def __init__(self, net, device, cfg):
        super().__init__(net, device, cfg)

class BBBTrainer(BaseTrainer):
    def __init__(self, net, device, out_channels, cfg):
        super().__init__(net, device, cfg)  # Initialize BaseTrainer
        self.out_channels = out_channels
        self.load_checkpoint()
        # Configuration object that includes num_samples, beta_type, etc.
        self.cfg = cfg
        self.history = {
            'loss_train': [], 'loss_valid': [],
            'acc_train': [], 'acc_valid': [],
            'kl_train': [], 'beta': [],
        }

    def train_model(self, data_loader, optimizer, mode='train'):
        self.net.train() if mode == 'train' else self.net.eval()
        num_ens = self.cfg.train_ens if mode == 'train' else self.cfg.valid_ens
        criterion = metrics.ELBO(self.cfg.num_samples).to(self.device)
        total_loss, total_kl, total_acc = 0.0, 0.0, 0.0
        num_batches = len(data_loader)
        beta_type = self.cfg.beta_type
        kl_list = []
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = torch.zeros(inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3], num_ens).to(self.device)
            kl = 0.0
            for j in range(num_ens):
                net_out = self.net(inputs)
                _kl = self.net.get_kl_loss_layers()
                kl += _kl
                outputs[:, :, :, :, j] = net_out                
            outputs = torch.mean(outputs, dim=4)
            kl = kl / num_ens
            kl_list.append(kl)
            beta = metrics.get_beta([], len(data_loader), beta_type, [], [])
            
            if mode == 'train':
                optimizer.zero_grad()
                loss, beta = criterion(outputs, labels, kl, beta)
                loss.backward()
                optimizer.step()
            else:
                loss, _ = criterion(outputs, labels, kl, 0)  # Beta is 0 for validation

            total_loss += loss.detach().item()
            total_acc += metrics.acc(outputs, labels).item()
        
        mean_kl = torch.mean(torch.stack(kl_list)).item() # Torch for tensors instead of numpy 
        self.update_history(mode, total_loss / num_batches, total_acc / num_batches, mean_kl)

        return total_loss / num_batches, total_acc / num_batches

    def update_history(self, mode, avg_loss, avg_acc, avg_kl):
        if mode == 'train':
            self.history['loss_train'].append(avg_loss)
            self.history['acc_train'].append(avg_acc)
            self.history['kl_train'].append(avg_kl)
        else:
            self.history['loss_valid'].append(avg_loss)
            self.history['acc_valid'].append(avg_acc)
        # Save history
        _, history_path, _ = construct_paths(self.cfg)
        np.savez(history_path, **self.history)

class HMCTrainer(BaseTrainer):
    def __init__(self, net, device, cfg):
        self.net = net.to(device)
        self.device = device
        self.cfg = cfg
        self.start_iter = int(os.environ.get('start_iter'))
        self.end_iter = int(os.environ.get('end_iter'))

    def _initialize_params(self):
        params_hmc_path = construct_paths(self.cfg)[2] + '/HMC_unet.pkl'
        if os.path.exists(params_hmc_path):
            return torch.load(params_hmc_path, map_location=self.device)[0]
        elif self.cfg.preload and not os.path.exists(params_hmc_path):
            state_dict = torch.load(construct_det_path(self.cfg), map_location=self.device)
            self.net.load_state_dict(state_dict)
        return hamiltorch.util.flatten(self.net).to(self.device).clone()
    
    def _save_max_indices(self, max_indices):
        save_path_indices = os.path.join(construct_paths(self.cfg)[2], 'max_indices.pkl')
        with open(save_path_indices, 'wb') as file:
            pickle.dump(max_indices, file)
    
    def _save_max_conv_indices(self, max_indices, layer_names):
        save_path = os.path.join(construct_paths(self.cfg)[2], 'max_conv_indices.pkl')
        max_indices_with_names = {'indices': max_indices, 'names': layer_names}
        with open(save_path, 'wb') as file:
            pickle.dump(max_indices_with_names, file)

    def _perform_hmc_sampling(self, params_init, max_indices, tau_list):
        test_set, test_loader, out_channels, X_ts, Y_ts, names_ts = load_test_data(self.cfg)
        X_ts_tensor = torch.tensor(X_ts, dtype=torch.float32)
        Y_ts_tensor = torch.tensor(Y_ts, dtype=torch.float32)
        X_ts = X_ts_tensor.to(self.device)
        Y_ts = Y_ts_tensor.to(self.device)
        start_iter = self.start_iter
        end_iter = self.end_iter    
        print(f"Start iteration: {start_iter}, End iteration: {end_iter}")
        _, train_loader, _, _, _, _, _, _, _, X_tr_tensor, Y_tr_tensor, _, _, _, _ = load_data(self.cfg)
        return hamiltorch.sample_model(
            self.net, X_tr_tensor.to(self.device), Y_tr_tensor.to(self.device),
            params_init=params_init, model_loss=metrics.mse_loss, num_samples=self.cfg.num_samples_hmc,
            num_samples_start=start_iter, num_samples_end=end_iter, burn=self.cfg.burn,
            step_size=self.cfg.step_size, num_steps_per_sample=self.cfg.L, tau_out=self.cfg.tau_out,
            tau_list=tau_list, normalizing_const=self.cfg.normalizing_const, store_on_GPU=self.cfg.store_on_GPU,
            sampler=hamiltorch.Sampler.HMC, weight_path=construct_paths(self.cfg)[0],
            X_batch=X_ts_tensor.to(self.device), Y_batch=Y_ts_tensor.to(self.device), net=self.net,
            data_path=self.cfg.data_path, tau=self.cfg.tau, cfg=self.cfg, max_indices=max_indices
        )
        
    def _save_params_hmc(self, params_hmc):
        params_hmc_path = construct_paths(self.cfg)[2] + '/params_HMC.pkl'
        torch.save(params_hmc, params_hmc_path)
        
    def run_training(self):
        self.net.train()
        # Prepare initial parameters for HMC
        params_init = self._initialize_params()
        flattened_params = hamiltorch.util.flatten(self.net).to(self.device)
        max_indices = get_largest_params(flattened_params,10)
        max_conv_indices, max_conv_indices_names = get_largest_conv_params(self.net, 20)   
        self._save_max_indices(max_indices)
        self._save_max_conv_indices(max_conv_indices,max_conv_indices_names)
        # Configuration for HMC sampling
        tau_list = comp_tau_list(self.net, self.cfg.tau, self.device)
        # HMC Sampling
        params_hmc = self._perform_hmc_sampling(params_init, max_indices, tau_list)
        # Save parameters after sampling
        self._save_params_hmc(params_hmc)

        return params_hmc

class Training:
    def __init__(self, cfg):
        self.device = cfg.device
        self.cfg = cfg
        drop_rate = 0 # !!! we want to study the effect of dropout only during inference time, not training time 
        self.net = self.select_network(
            cfg.method, cfg.nfilters, cfg.kernel_size, cfg.layer_type, drop_rate).to(cfg.device)
        
    def select_network(self, network_type, nfilters, kernel_size, layer_type, drop_rate):

        if network_type in ['BBB', 'BBB_LRT']:
            return BayesByBackprop(nfilters, kernel_size, layer_type)
        elif network_type in ['Deterministic', 'HMC']:
            return Deterministic(nfilters, kernel_size)
        elif network_type in ['MCD']:
            return MonteCarloDropout(nfilters, kernel_size, drop_rate, self.cfg.drop_idx_en, self.cfg.drop_idx_dec)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def select_trainer(self, method, out_channels):
        if method == 'BBB' or method == 'BBB_LRT':
            return BBBTrainer(self.net, self.device, out_channels, self.cfg)
        elif method == 'Deterministic':
            return DeterministicTrainer(self.net, self.device, self.cfg)
        elif method == 'MCD':
            return MCDTrainer(self.net, self.device, self.cfg)
        else:
            return HMCTrainer(self.net, self.device, self.cfg)

    def run_train(self):
        cfg = self.cfg
        device = cfg.device
        out_channels = load_data(cfg)[3]        
        trainer = self.select_trainer(cfg.method, out_channels)
        trainer.run_training()