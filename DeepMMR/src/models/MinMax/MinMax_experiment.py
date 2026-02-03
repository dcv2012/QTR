import os, csv
import numpy as np
import torch
from pathlib import Path

from src.data.simulation import generate_train_simulation_h, generate_test_simulation_h
from src.models.MinMax.proxITR import proxITR

def run_minmax_experiments(num_runs: int, scenario: str, n_train: int, n_test: int, output_dir: str):
    path = Path(output_dir)
    
    if not path.exists():
        os.makedirs(path)
    
    results = []
    
    for i in range(num_runs):
        random_seed = np.random.randint(0, 2**10 - 1)
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
        train_data = generate_train_simulation_h(n_train, scenario)
        test_data  = generate_test_simulation_h(n_test, scenario)
        
        rhos = np.power(2.,-np.arange(-2,7))
        proxy = proxITR(A=train_data.treatment, X=train_data.backdoor, Z=train_data.treatment_proxy, W=train_data.outcome_proxy, Y=train_data.outcome, learning_rate= 0.1, n_epoch=2000, opt='LBFGS', verbose=True, qtl = 0.4)

        predict_q0, predict_q1 = proxy.predict_q0(n_components=int(2*np.sqrt(n_train)),index='all')
        
        data_ZX = np.concatenate((test_data.treatment_proxy, test_data.backdoor), axis=1)
        pred_q0, pred_q1 = predict_q0(data_ZX), predict_q1(data_ZX)
        
        res1 = test_data.outcome.reshape(-1) * pred_q1 * (test_data.treatment.reshape(-1) == np.ones(test_data.treatment.shape[0]))
        res0 = test_data.outcome.reshape(-1) * pred_q0 * (test_data.treatment.reshape(-1) == np.zeros(test_data.treatment.shape[0]))
        res = [res1.mean(), res0.mean()]
        results.append(['MinMax'] + res)
    
    file_path = path.joinpath(f"minmax_{scenario}_{n_train}.csv")
    
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['Method', 'E[Y(1)]', 'E[Y(0)]'])
        writer.writerows(results)