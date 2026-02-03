import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from src.data.data_class import MMRTrainDataSetTorch_h, MMRTestDataSet_h, MMRTrainDataSetTorch_q, MMRTestDataSetTorch_q
from src.models.MMR.MMR_loss import MMR_loss
from src.models.MMR.MMR_model import MLP_for_MMR
from src.models.MMR.kernel_utils import calculate_kernel_matrix


class EarlyStopping:
    def __init__(self, patience=20, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class MMR_Trainer_Simulation:
    def __init__(self, train_params: Dict[str, Any], random_seed: int):
        self.train_params = train_params
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.gpu_flg = torch.cuda.is_available()
        self.l2_penalty = train_params['l2_penalty']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']


    def compute_kernel(self, kernel_inputs):
        return calculate_kernel_matrix(kernel_inputs)
    
    def train(self, train_t: MMRTrainDataSetTorch_q, val_t: MMRTrainDataSetTorch_q):
        input_size = 1 + train_t.backdoor.shape[1]
        model = MLP_for_MMR(input_dim=input_size, train_params=self.train_params)
    
        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model.cuda()
    
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                   patience=10, threshold=1e-4)
   

        train_a, train_w, train_x, train_y, train_z, train_tt = train_t.treatment, train_t.outcome_proxy, train_t.backdoor, train_t.outcome, train_t.treatment_proxy, train_t.treatment_target
        val_a, val_w, val_x, val_y, val_z, val_tt = val_t.treatment, val_t.outcome_proxy, val_t.backdoor, val_t.outcome, val_t.treatment_proxy, val_t.treatment_target
    
        # batch training
        train_dataset = TensorDataset(train_a, train_w, train_x, train_y, train_z, train_tt)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
        early_stopping = EarlyStopping(patience=20,delta=1e-4)
        best_model_state = None
        best_val_loss = float('inf')
    
        for epoch in tqdm(range(self.n_epochs)):
            # train
            model.train()
            total_loss = 0
    
            for batch_a, batch_w, batch_x, batch_y, batch_z, batch_tt in train_loader:
                if self.gpu_flg:
                    batch_a, batch_w, batch_x, batch_y, batch_z, batch_tt = batch_a.cuda(), batch_w.cuda(), batch_x.cuda(), batch_y.cuda(), batch_z.cuda(), batch_tt.cuda()
    
                optimizer.zero_grad()
                batch_inputs = torch.cat((batch_z, batch_x), dim=1)
                pred = model(batch_inputs)
    
                kernel_inputs = torch.cat((batch_w, batch_x), dim=1)
                kernel_matrix = self.compute_kernel(kernel_inputs)
    
                loss = torch.abs(MMR_loss(pred * batch_tt, torch.ones_like(batch_y), kernel_matrix, self.loss_name))
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
    
            # eval
            model.eval()
            with torch.no_grad():
                val_inputs = torch.cat((val_z, val_x), dim=1)
                val_pred = model(val_inputs)
    
                val_kernel_inputs = torch.cat((val_w, val_x), dim=1)
                val_kernel_matrix = self.compute_kernel(val_kernel_inputs)
    
                val_loss = abs(MMR_loss(val_pred * val_tt, torch.ones_like(val_y), val_kernel_matrix, self.loss_name).detach().cpu().item())
    
            scheduler.step(val_loss)
    
            # early stop
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
    
            # update the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
    
        # load the best model
        model.load_state_dict(best_model_state)
    
        return best_val_loss, model


    @staticmethod
    def predict(model, test_data_t: MMRTestDataSetTorch_q):
        device = next(model.parameters()).device  # 获取模型设备
        
        intervention_array_len = 1
        n_samples = test_data_t.treatment_proxy.shape[0]
        tempZ = test_data_t.treatment_proxy.to(device) #to
        tempX = test_data_t.backdoor.to(device) #to
        model_inputs_test = torch.cat((tempZ, tempX),dim = 1)

        with torch.no_grad():
            E_zx = model(model_inputs_test)

        return E_zx.cpu()


class MMR_Trainer_RHC:
    def __init__(self, train_params: Dict[str, Any], random_seed: int):
        self.train_params = train_params
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.gpu_flg = torch.cuda.is_available()
        self.l2_penalty = train_params['l2_penalty']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']


    def compute_kernel(self, kernel_inputs):
        return calculate_kernel_matrix(kernel_inputs)
    
    def train(self, train_t: MMRTrainDataSetTorch_q, val_t: MMRTrainDataSetTorch_q):
        input_size = 2 + train_t.backdoor.shape[1]
        model = MLP_for_MMR(input_dim=input_size, train_params=self.train_params)
    
        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model.cuda()
    
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                   patience=10, threshold=1e-4)
   
    
        train_a, train_w, train_x, train_y, train_z, train_tt = train_t.treatment, train_t.outcome_proxy, train_t.backdoor, train_t.outcome, train_t.treatment_proxy, train_t.treatment_target
        val_a, val_w, val_x, val_y, val_z, val_tt = val_t.treatment, val_t.outcome_proxy, val_t.backdoor, val_t.outcome, val_t.treatment_proxy, val_t.treatment_target
    
        # batch training
        train_dataset = TensorDataset(train_a, train_w, train_x, train_y, train_z, train_tt)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
        early_stopping = EarlyStopping(patience=10, delta=1e-4)
        best_model_state = None
        best_val_loss = float('inf')
    
        for epoch in tqdm(range(self.n_epochs)):
            # train
            model.train()
            total_loss = 0
    
            for batch_a, batch_w, batch_x, batch_y, batch_z, batch_tt in train_loader:
                if self.gpu_flg:
                    batch_a, batch_w, batch_x, batch_y, batch_z, batch_tt = batch_a.cuda(), batch_w.cuda(), batch_x.cuda(), batch_y.cuda(), batch_z.cuda(), batch_tt.cuda()
    
                optimizer.zero_grad()
                batch_inputs = torch.cat((batch_z, batch_x), dim=1)
                pred = model(batch_inputs)
    
                kernel_inputs = torch.cat((batch_w, batch_x), dim=1)
                kernel_matrix = self.compute_kernel(kernel_inputs)
    
                loss = torch.abs(MMR_loss(pred * batch_tt, torch.ones_like(batch_y), kernel_matrix, self.loss_name))
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
    
            # eval
            model.eval()
            with torch.no_grad():
                val_inputs = torch.cat((val_z, val_x), dim=1)
                val_pred = model(val_inputs)
    
                val_kernel_inputs = torch.cat((val_w, val_x), dim=1)
                val_kernel_matrix = self.compute_kernel(val_kernel_inputs)
    
                val_loss = abs(MMR_loss(val_pred * val_tt, torch.ones_like(val_y), val_kernel_matrix, self.loss_name).detach().cpu().item())
    
            scheduler.step(val_loss)
    
            # early stop
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
    
            # update the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
    
        # load the best model
        model.load_state_dict(best_model_state)
    
        return best_val_loss, model


    @staticmethod
    def predict(model, test_data_t: MMRTestDataSetTorch_q):
        intervention_array_len = 1
        n_samples = test_data_t.treatment_proxy.shape[0]
        tempZ = test_data_t.treatment_proxy
        tempX = test_data_t.backdoor
        model_inputs_test = torch.cat((tempZ, tempX),dim = 1)

        with torch.no_grad():
            E_zx = model(model_inputs_test)

        return E_zx.cpu()

