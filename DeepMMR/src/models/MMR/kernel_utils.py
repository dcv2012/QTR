import numpy as np
import torch

import torch

def calculate_kernel_matrix(dataset, **kwargs):
    
    K_X_euclidean = torch.cdist(dataset, dataset, p=2) ** 2
    triuInd = torch.triu_indices(K_X_euclidean.size(0),K_X_euclidean.size(0),offset=1)
    K_X_euclidean_upper = K_X_euclidean[triuInd[0],triuInd[1]]
    gamma = 1./torch.quantile(K_X_euclidean_upper, 0.5)
    return  torch.exp(-gamma * K_X_euclidean).squeeze()


def calculate_kernel_matrix_batched(dataset, batch_indices: tuple, **kwargs):
    start, end = batch_indices
    x = dataset[start:end]
    y = dataset
    dists = torch.cdist(x, y) ** 2
    gamma = 1 / torch.median(dists[dists != 0])
    return torch.exp(-gamma * dists).squeeze()