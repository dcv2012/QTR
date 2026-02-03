import os
import csv
import numpy as np
import torch
from pathlib import Path

from src.data.simulation import generate_train_simulation_h, generate_test_simulation_h
from src.models.GMM.GMM import GMM_q, predict_q


def run_gmm_experiments(num_runs: int, scenario: str, n_train: int, n_test: int, output_dir: str):
    path = Path(output_dir)
    
    if not path.exists():
        os.makedirs(path)

    results = []
    
    for _ in range(num_runs):
        random_seed = np.random.randint(0, 2**15 - 1)
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
        train_data = generate_train_simulation_h(n_train, scenario)
        test_data  = generate_test_simulation_h(n_test, scenario)

        t_est = GMM_q(train_data)
        
        pred_q1 = predict_q(t_est, np.ones_like(test_data.treatment), test_data.treatment_proxy, test_data.backdoor).flatten()
        pred_q0 = predict_q(t_est, np.zeros_like(test_data.treatment), test_data.treatment_proxy, test_data.backdoor).flatten()
        
        res1 = test_data.outcome.flatten() * pred_q1 * (test_data.treatment.flatten() == 1)
        res0 = test_data.outcome.flatten() * pred_q0 * (test_data.treatment.flatten() == 0)
        res = [res1.mean(), res0.mean()]
        results.append(['GMM'] + res)
    
    file_path = path.joinpath(f"gmm_{scenario}_{n_train}.csv")
    
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['Method', 'E[Y(1)]', 'E[Y(0)]'])
        writer.writerows(results)
