import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal, uniform, randint

from src.data.data_class import MMRTrainDataSet_q, MMRTestDataSet_q
from data_generate import origin_para_set, adjust_para_set_for_new_coding, data_gen, intervened_data_gen

# get new parameter set for A in {-1,1}
new_para_set = adjust_para_set_for_new_coding(origin_para_set)

def generate_data(n_sample: int, scenario: str = 'S1'):
    data = data_gen(n_sample, new_para_set)
    Y0_arr = np.array(data['Y0'])   # X
    A1_arr = np.array(data['A1'])
    W1_arr = np.array(data['W1'])
    Z1_arr= np.array(data['Z1'])
    Y1_arr = np.array(data['Y1'])
    
    intervened_data_1p = intervened_data_gen(n_sample, new_para_set, [1,1])
    intervened_data_1n = intervened_data_gen(n_sample, new_para_set, [-1,1])
    Y1p_arr_= np.array(intervened_data_1p['Y1'])
    Y1n_arr_= np.array(intervened_data_1n['Y1'])
    Ep = Y1p_arr_.mean()
    En = Y1n_arr_.mean()
    
    return (A1_arr.reshape(-1, 1), 
            W1_arr.reshape(-1, 1), 
            Y0_arr.reshape(-1, 1),
            Y1_arr.reshape(-1, 1),
            Z1_arr.reshape(-1, 1),
            Ep,En)



# Generate training data
def generate_train_simulation_q(n_sample: int, scenario: str = 'S1', **kwargs):
    A1, W1, Y0, Y1, Z1, _, _ = generate_data(n_sample, scenario=scenario)
    print(Y0.shape)
    A1_target = np.zeros(n_sample) # all set to -1
    return MMRTrainDataSet_q(treatment=A1,
                             treatment_target=A1_target,
                             treatment_proxy=Z1,
                             outcome_proxy=W1,
                             outcome=Y1,
                             backdoor=Y0)

# Generate test data
def generate_test_simulation_q(n_sample: int, scenario: str, **kwargs):
    A1, W1, Y0, Y1, Z1, E1, E0 = generate_data(n_sample, scenario=scenario)
    return MMRTestDataSet_q(treatment=A1,
                           treatment_proxy=Z1,
                           outcome_proxy=W1,
                           outcome=Y1,
                           backdoor=Y0,
                           structural=[E1, E0])
    

if __name__ == "__main__":
    # A1, W1, Y0, Y1, Z1, EP, EN = generate_data(100000, scenario='S1')
    mtd = generate_train_simulation_q(1000, scenario='S1')
    print(type(mtd))
    