import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data.data_class import MMRTestDataSet_h, MMRTrainDataSetTorch_h,MMRTestDataSetTorch_h
from src.data.data_class import MMRTrainDataSet_q, MMRTrainDataSetTorch_q, MMRTestDataSetTorch_q
from src.data.simulation import generate_test_simulation_q, generate_train_simulation_h


def GMM_q(train_data):
    A = train_data.treatment
    Z = train_data.treatment_proxy
    W = train_data.outcome_proxy
    X = train_data.backdoor
    Y = train_data.outcome
    
    
    def q(A, Z, X, t):
        exponent = (-1)**(1 - A) * (t[0] + t[1] * A + t[2] * Z + np.sum(t[3:] * X, axis=1, keepdims=True))
        return 1 + np.exp(exponent)
        
    # Define the objective function of GMM
    def gmm_objective(t, A, W, X, Z, Y):
        residuals = (-1)**(1 - A) * q(A, Z, X, t)
        constant_vector = np.array([0, 0, 1, 0, 0])
        instruments = np.hstack([np.ones_like(Z), W, A, X])   #  Instrumental Variables (1, A, W, X)
        moment_conditions = residuals * instruments - constant_vector
        return np.mean(moment_conditions, axis=0)
    
    # Define the minimized objective function: the square residual of GMM
    def gmm_loss(t, A, W, X, Z, Y):
       moment_conditions = gmm_objective(t, A, W, X, Z, Y)
       return np.sum(np.abs(moment_conditions))

    # Initial parameters
    t_init = np.random.randint(0, 3, 5)
    
    # Fit the parameters using the minimization function of scipy
    result = minimize(gmm_loss, t_init, args=(A, W, X, Z, Y), method = 'CG')
    t_estimate = result.x
    
    return t_estimate

# predict function
# Use the t obtained from training to predict q(A, Z, X)
def predict_q(t_estimate, A, Z, X):
    def q(A, Z, X, t):
        exponent = (-1)**(1 - A) * (t[0] + t[1] * A + t[2] * Z + np.sum(t[3:] * X, axis=1, keepdims=True))
        return 1 + np.exp(exponent)
    
    predictions = q(A, Z, X, t_estimate)
    return predictions



def GMM_q_rhc(train_data):
    A = train_data.treatment
    Z = train_data.treatment_proxy
    W = train_data.outcome_proxy
    X = train_data.backdoor
    Y = train_data.outcome
    
    
    def q(A, Z, X, t):
        exponent = (-1)**(1 - A) * (t[0] + t[1] * A + np.sum(t[2:4] * Z, axis=1, keepdims=True) *  + np.sum(t[4:] * X, axis=1, keepdims=True))
        return 1 + np.exp(exponent)
        
    # Define the objective function of GMM
    def gmm_objective(t, A, W, X, Z, Y):
        residuals = (-1)**(1 - A) * q(A, Z, X, t)
        constant_vector = np.concatenate((np.array([0, 0, 0, 1,]), np.zeros(20,)))
        instruments = np.hstack([np.ones_like(A), W, A, X])
        moment_conditions = residuals * instruments - constant_vector
        return np.mean(moment_conditions, axis=0)
    
    # Define the minimized objective function: the square residual of GMM
    def gmm_loss(t, A, W, X, Z, Y):
        moment_conditions = gmm_objective(t, A, W, X, Z, Y)
        return np.sum(moment_conditions**2)
    
    # Initial parameters
    t_init = np.random.rand(24)
    
    # Fit the parameters using the minimization function of scipy
    result = minimize(gmm_loss, t_init, args=(A, W, X, Z, Y), method='BFGS', options={'maxiter': 1000})
    t_estimate = result.x
    
    return t_estimate

# predict function
# Use the t obtained from training to predict q(A, Z, X)
def predict_q_rhc(t_estimate, A, Z, X):
    def q(A, Z, X, t):
        exponent = (-1)**(1 - A) * (t[0] + t[1] * A + np.sum(t[2:4] * Z, axis=1, keepdims=True) *  + np.sum(t[4:] * X, axis=1, keepdims=True))
        return 1 + np.exp(exponent)
    
    predictions = q(A, Z, X, t_estimate)
    return predictions