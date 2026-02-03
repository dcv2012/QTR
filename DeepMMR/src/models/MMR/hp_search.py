import numpy as np
import pandas as pd
import json
import itertools
import torch
from pathlib import Path
from typing import Dict, Any
from skopt import gp_minimize
from skopt.space import Real, Integer

from src.data.data_class import MMRTrainDataSet_h, MMRTestDataSet_h, MMRTrainDataSetTorch_h, MMRTestDataSetTorch_h
from src.data.data_class import MMRTrainDataSet_q, MMRTestDataSet_q, MMRTrainDataSetTorch_q, MMRTestDataSetTorch_q
from src.data.simulation import generate_train_simulation_q
from src.data.rhc import generate_train_rhc, generate_val_rhc, generate_test_rhc
from src.models.MMR.MMR_trainers import MMR_Trainer_Simulation, MMR_Trainer_RHC



def hp_search_simulation(num_runs: int, scenario: str, kind: str, treatment: int, n_train: int, n_test: int, output_dir: str):
    
    random_seed = np.random.randint(0, 2**16 - 1)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
        
    # Generate data
    train_data = generate_train_simulation_q(int(n_train * 0.75), scenario)
    val_data = generate_train_simulation_q(int(n_train * 0.25), scenario)
    test_data = generate_train_simulation_q(n_test, scenario)
    
    if treatment == 1:
        train_data = MMRTrainDataSet_q(
            treatment=train_data.treatment,
            treatment_target=(train_data.treatment == 1), 
            treatment_proxy=train_data.treatment_proxy,
            outcome_proxy=train_data.outcome_proxy,
            outcome=train_data.outcome,
            backdoor=train_data.backdoor
        )
        train_data_t = MMRTrainDataSetTorch_q.from_numpy(train_data)
        
        val_data = MMRTrainDataSet_q(
            treatment=val_data.treatment,
            treatment_target=(val_data.treatment == 1),
            treatment_proxy=val_data.treatment_proxy,
            outcome_proxy=val_data.outcome_proxy,
            outcome=val_data.outcome,
            backdoor=val_data.backdoor
        )
        val_data_t = MMRTrainDataSetTorch_q.from_numpy(val_data)
    elif treatment == -1: #-1
        train_data = MMRTrainDataSet_q(
            treatment=train_data.treatment,
            treatment_target=(train_data.treatment == -1),
            treatment_proxy=train_data.treatment_proxy,
            outcome_proxy=train_data.outcome_proxy,
            outcome=train_data.outcome,
            backdoor=train_data.backdoor
        )
        train_data_t = MMRTrainDataSetTorch_q.from_numpy(train_data)
        
        val_data = MMRTrainDataSet_q(
            treatment=val_data.treatment,
            treatment_target=(val_data.treatment == 0),
            treatment_proxy=val_data.treatment_proxy,
            outcome_proxy=val_data.outcome_proxy,
            outcome=val_data.outcome,
            backdoor=val_data.backdoor
        )
        val_data_t = MMRTrainDataSetTorch_q.from_numpy(val_data)
    
    
    # Define the hyperparameter space
    param_space = [
        Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
        Real(1e-6, 1e-3, prior='log-uniform', name='l2_penalty'),
        Real(0.1, 0.5, prior='uniform', name='dropout_prob'),
        Integer(3, 6, name='network_depth'),
        Integer(20, 40, name='network_width')
    ]
    keys = ['learning_rate', 'l2_penalty', 'dropout_prob', 'network_depth', 'network_width']

    def evaluate_model(params):
        model_config = {
            "n_epochs": 200,
            "batch_size": 100,
            "loss_name": f"{kind.upper()}_statistic"
        }
        model_config.update(dict(zip(keys, params)))
        
        trainer = MMR_Trainer_Simulation(model_config, random_seed)
        loss, _ = trainer.train(train_data_t, val_data_t)
        
        return abs(loss)
    
    # Optimize
    best_params = gp_minimize(
        evaluate_model,
        param_space,
        n_calls=num_runs,
        random_state=random_seed
    )

    # optimal hyperparameters
    best_params_converted = {key: float(value) for key, value in zip(keys, best_params.x)}
    best_params_converted.update({
        "n_epochs": 200,
        "batch_size": 100,
        "loss_name": f"{kind.upper()}_statistic"
    })

    # output file path
    output_file_path = output_dir / f'mmr_{scenario.lower()}_{kind.lower()}_{treatment}.json'
    with open(output_file_path, 'w') as f:
        json.dump(best_params_converted, f, indent=4)



def hp_search_rhc(num_runs: int, kind: str, treatment: int, output_dir: str):
    
    random_seed = np.random.randint(0, 2**16 - 1)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
        
    # Generate data
    train_data = generate_train_rhc(False)
    val_data = generate_val_rhc(False)
    test_data = generate_test_rhc(False)
    
    if treatment == 1:
        train_data = MMRTrainDataSet_q(
            treatment=train_data.treatment,
            treatment_target=(train_data.treatment == 1),
            treatment_proxy=train_data.treatment_proxy,
            outcome_proxy=train_data.outcome_proxy,
            outcome=train_data.outcome,
            backdoor=train_data.backdoor
        )
        train_data_t = MMRTrainDataSetTorch_q.from_numpy(train_data)
        
        val_data = MMRTrainDataSet_q(
            treatment=val_data.treatment,
            treatment_target=(val_data.treatment == 1),
            treatment_proxy=val_data.treatment_proxy,
            outcome_proxy=val_data.outcome_proxy,
            outcome=val_data.outcome,
            backdoor=val_data.backdoor
        )
        val_data_t = MMRTrainDataSetTorch_q.from_numpy(val_data)
        
    elif treatment == 0:
        train_data = MMRTrainDataSet_q(
            treatment=train_data.treatment,
            treatment_target=(train_data.treatment == 0),
            treatment_proxy=train_data.treatment_proxy,
            outcome_proxy=train_data.outcome_proxy,
            outcome=train_data.outcome,
            backdoor=train_data.backdoor
        )
        train_data_t = MMRTrainDataSetTorch_q.from_numpy(train_data)
        
        val_data = MMRTrainDataSet_q(
            treatment=val_data.treatment,
            treatment_target=(val_data.treatment == 0),
            treatment_proxy=val_data.treatment_proxy,
            outcome_proxy=val_data.outcome_proxy,
            outcome=val_data.outcome,
            backdoor=val_data.backdoor
        )
        val_data_t = MMRTrainDataSetTorch_q.from_numpy(val_data)
        
    # Define the hyperparameter space
    param_space = [
        Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
        Real(1e-6, 1e-3, prior='log-uniform', name='l2_penalty'),
        Real(0.1, 0.5, prior='uniform', name='dropout_prob'),
        Integer(4, 9, name='network_depth'),
        Integer(20, 50, name='network_width')
    ]
    keys = ['learning_rate', 'l2_penalty', 'dropout_prob', 'network_depth', 'network_width']

    def evaluate_model(params):
        model_config = {
            "n_epochs": 200,
            "batch_size": 100,
            "loss_name": f"{kind.upper()}_statistic"
        }
        model_config.update(dict(zip(keys, params)))
        
        trainer = MMR_Trainer_RHC(model_config, random_seed)
        loss, _ = trainer.train(train_data_t, val_data_t)
        return loss
    
    # optimize
    best_params = gp_minimize(
        evaluate_model,
        param_space,
        n_calls=num_runs,
        random_state=random_seed
    )

    # optimal hyperparameters
    best_params_converted = {key: float(value) for key, value in zip(keys, best_params.x)}
    best_params_converted.update({
        "n_epochs": 200,
        "batch_size": 100,
        "loss_name": f"{kind.upper()}_statistic"
    })

    # output file path
    output_file_path = output_dir / f'mmr_rhc_{kind.lower()}_{treatment}.json'
    with open(output_file_path, 'w') as f:
        json.dump(best_params_converted, f, indent=4)
