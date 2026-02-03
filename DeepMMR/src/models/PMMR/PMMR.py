import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import csv

from src.utils.kernel_func import ColumnWiseGaussianKernel, AbsKernel, BinaryKernel, GaussianKernel
from src.data.preprocess import get_preprocessor_ate
from src.data.data_class import MMRTrainDataSet_h, MMRTestDataSet_h
from src.data.simulation import generate_train_simulation_h, generate_test_simulation_h

def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel, AbsKernel, AbsKernel]:
    if data_name == "simulation":
        return BinaryKernel(), GaussianKernel(), GaussianKernel(), GaussianKernel()
    elif data_name == "rhc":
        return BinaryKernel(), GaussianKernel(), GaussianKernel(), GaussianKernel()
    else:
        return ColumnWiseGaussianKernel(), ColumnWiseGaussianKernel(), ColumnWiseGaussianKernel(), ColumnWiseGaussianKernel()


class PMMRModel_q:
    treatment_kernel_func: AbsKernel
    treatment_proxy_kernel_func: AbsKernel
    _proxy_kernel_func: AbsKernel
    backdoor_kernel_func: AbsKernel

    alpha: np.ndarray
    x_mean_vec: Optional[np.ndarray]
    w_mean_vec: np.ndarray
    train_treatment: np.ndarray
    train_treatment_proxy: np.ndarray

    def __init__(self, lam1, lam2=0.0001, scale=1.0, **kwargs):
        self.lam1 = lam1
        self.lam2 = lam2
        self.scale = scale
        self.x_mean_vec = None

    def fit(self, train_data: MMRTrainDataSet_h, data_name: str):
        kernels = get_kernel_func(data_name)
        self.treatment_kernel_func = kernels[0]
        self.treatment_proxy_kernel_func = kernels[1]
        self.outcome_proxy_kernel_func = kernels[2]
        self.backdoor_kernel_func = kernels[3]
        n_train = train_data.treatment.shape[0]
        
        
        index_1 = np.where(train_data.treatment == 1)[0]
        index_0 = np.where(train_data.treatment == 0)[0]
        self.permutation = np.concatenate((index_1, index_0))
        self.pos = index_1.shape[0]
        self.neg = index_0.shape[0]
        
        self.treatment_proxy_kernel_func.fit(train_data.treatment_proxy[self.permutation], scale=self.scale)
        self.treatment_kernel_func.fit(train_data.treatment[self.permutation], scale=self.scale)
        self.outcome_proxy_kernel_func.fit(train_data.outcome_proxy[self.permutation], scale=self.scale)

        if train_data.backdoor is not None:
            self.backdoor_kernel_func.fit(train_data.backdoor[self.permutation], scale=self.scale)

        treatment_mat = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment[self.permutation],
                                                                  train_data.treatment[self.permutation])
        treatment_proxy_mat = self.treatment_proxy_kernel_func.cal_kernel_mat(train_data.treatment_proxy[self.permutation],
                                                                              train_data.treatment_proxy[self.permutation])
        outcome_proxy_mat = self.outcome_proxy_kernel_func.cal_kernel_mat(train_data.outcome_proxy[self.permutation],
                                                                          train_data.outcome_proxy[self.permutation])
        backdoor_mat = np.ones((n_train, n_train))
        if train_data.backdoor is not None:
            backdoor_mat = self.backdoor_kernel_func.cal_kernel_mat(train_data.backdoor[self.permutation],
                                                                    train_data.backdoor[self.permutation])
            self.x_mean_vec = np.mean(backdoor_mat, axis=0)[:, np.newaxis]
        W1 = treatment_mat[:self.pos, :self.pos] * treatment_proxy_mat[:self.pos, :self.pos] * backdoor_mat[:self.pos, :self.pos]
        L1 = treatment_mat[:self.pos, :self.pos] * outcome_proxy_mat[:self.pos, :self.pos] * backdoor_mat[:self.pos, :self.pos]
        K1 = L1 @ W1 @ L1 + self.lam1 * n_train * L1 + self.lam2 * n_train * np.eye(self.pos)
        b1 = L1 @ W1 @ np.ones(self.pos)
        self.alpha1 = np.linalg.solve(K1, b1)
        
        W2 = treatment_mat[self.pos:, self.pos:] * treatment_proxy_mat[self.pos:, self.pos:] * backdoor_mat[self.pos:, self.pos:]
        L2 = treatment_mat[self.pos:, self.pos:] * outcome_proxy_mat[self.pos:, self.pos:] * backdoor_mat[self.pos:, self.pos:]
        K2 = L2 @ W2 @ L2 + self.lam1 * n_train * L2 + self.lam2 * n_train * np.eye(self.neg)
        b2 = L2 @ W2 @ np.ones(self.neg)
        self.alpha2 = np.linalg.solve(K2, b2)
        
        self.w_mean_vec = np.mean(outcome_proxy_mat, axis=0)[:, np.newaxis]
        self.train_treatment = train_data.treatment[self.permutation]
        self.train_backdoor = train_data.backdoor[self.permutation]
        self.train_treatment_proxy = train_data.treatment_proxy[self.permutation]

    def predict(self, treatment: np.ndarray) -> np.ndarray:
        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, treatment)
        test_kernel *= self.w_mean_vec
        if self.x_mean_vec is not None:
            test_kernel = test_kernel * self.x_mean_vec

        pred = self.alpha.T @ test_kernel
        return pred.T

    def predict_bridge1(self, treatment: np.ndarray, backdoor, treatment_proxy: np.ndarray) -> np.ndarray:
        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment[:self.pos], treatment)
        test_kernel *= self.backdoor_kernel_func.cal_kernel_mat(self.train_backdoor[:self.pos], backdoor)
        test_kernel *= self.treatment_proxy_kernel_func.cal_kernel_mat(self.train_treatment_proxy[:self.pos], treatment_proxy)

        pred = self.alpha1.T @ test_kernel
        return pred.T
    
    def predict_bridge0(self, treatment: np.ndarray, backdoor, treatment_proxy: np.ndarray) -> np.ndarray:
        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment[self.pos:], treatment)
        test_kernel *= self.backdoor_kernel_func.cal_kernel_mat(self.train_backdoor[self.pos:], backdoor)
        test_kernel *= self.treatment_proxy_kernel_func.cal_kernel_mat(self.train_treatment_proxy[self.pos:], treatment_proxy)

        pred = self.alpha2.T @ test_kernel
        return pred.T

    def evaluate(self, test_data: MMRTestDataSet_h):
        pred = self.predict(treatment=test_data.treatment)
        return np.mean((pred - test_data.structural) ** 2)

def pmmr_experiments_q(scenario, n_train, n_test, path, random_seed: int = 42):
    model_config = {"name": "pmmr", "lam1": 0.01, "lam2": 0.01, "scale": 0.5, "data_scaling": True}
    train_data_org = generate_train_simulation_h(n_train, scenario)
    test_data_org = generate_test_simulation_h(n_test, scenario)

    preprocessor = get_preprocessor_ate("Identity")
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    model = PMMRModel_q(**model_config)
    model.fit(train_data, "simulation")
    index1 = np.where(test_data.treatment == 1)[0]
    index2 = np.where(test_data.treatment == 0)[0]
    pred1 = model.predict_bridge1(np.ones((index1.shape[0],1)), test_data.backdoor[index1], test_data.treatment_proxy[index1])
    pred0 = model.predict_bridge0(np.zeros((index2.shape[0],1)), test_data.backdoor[index2], test_data.treatment_proxy[index2])
    res1 = test_data.outcome[index1].squeeze(-1) * pred1
    res0 = test_data.outcome[index2].squeeze(-1) * pred0
    
    pred_q = np.zeros((len(test_data.outcome),))
    pred_q[index1] = pred1
    pred_q[index2] = pred0
    
    
    res = [res1.mean(), res0.mean()]
    file_path = path.joinpath(f"pmmr_{scenario}_{n_train}.csv")
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PMMR'] + res)