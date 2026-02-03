import os
import numpy as np
from pathlib import Path
from src.models.PMMR.PMMR import pmmr_experiments_q


def run_pmmr_experiments(num_runs: int, scenario: str, n_train: int, n_test: int, output_dir: str):
    path = Path(output_dir)
    
    if not path.exists():
        os.makedirs(path)
    
    for _ in range(num_runs):
        rand_seed = np.random.randint(0, 2**16 - 1)
        np.random.seed(rand_seed)
        pmmr_experiments_q(scenario, n_train, n_test, path, rand_seed)
