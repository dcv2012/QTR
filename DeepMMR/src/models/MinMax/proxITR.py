import numpy as np
import pandas as pd
from src.models.MinMax.rkhs_scaler import RKHSIV, RKHSIVCV, ApproxRKHSIV, ApproxRKHSIVCV, RKHSIV_q, RKHSIVCV_q, ApproxRKHSIV_q, ApproxRKHSIVCV_q
import sklearn
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn import pipeline
from sklearn.preprocessing import RobustScaler as Scaler
#from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from src.models.MinMax.torchSVC import RegimeLearner
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats

#from copy import deepcopy

def _check_all(param):
    return (isinstance(param, str) and (param == 'all'))



class ApproxNonLinear:
    def __init__(self, transX, featX, rho = 0.1, n_epoch = 2000, batch_size = 200, learning_rate=0.1, opt = 'LBFGS'):
        """
        Parameters:
            rho : the penalty coefficient
        """
        self.transX = transX  # a fitted transformer
        self.featX = featX      # Nystroem features
        self.rho = rho
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.opt = opt
    
    def fit(self, X, y, sample_weight):
        # Standardize X
        X = self.transX.transform(X)
        # Kernel Approximation
        X_transformed = self.featX.transform(X)
        self.regime = RegimeLearner(n_epoch=self.n_epoch, batch_size = self.batch_size, rho = self.rho, learning_rate=self.learning_rate, opt=self.opt)\
            .fit(X=X_transformed, y=y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X = self.transX.transform(X)
        X_transformed = self.featX.transform(X)
        return self.regime.predict(X=X_transformed)

def FastOptBW(X, y, sample_weight=None, n_gammas=10):
    """
    Fast Optimal Bandwidth Selection for RBF Kernel using HSIC (Damodaran 2018, IEEE)
    Input: 
        X: covariates of SVM
        y: response of SVM: {-1,1}
        sample_weight: for weight w_i on sample (X_i,y_i), a posiitive np vector with sum=1
        n_gammas: number of gammas to try
    Output: 
        Optimal gamma for the RBF kernel in terms of classification
    """
    # Check cuda GPU device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    X = torch.tensor(X, dtype=torch.float32, device=device)
    K_X_euclidean = torch.square(torch.cdist(X,X))
    triuInd = torch.triu_indices(K_X_euclidean.size(0),K_X_euclidean.size(0),offset=1)
    K_X_euclidean_upper = K_X_euclidean[triuInd[0],triuInd[1]]
    gammas = 1./torch.quantile(K_X_euclidean_upper,
                                torch.linspace(0.1, 0.9, n_gammas, device=device))

    reci_pos = 2./(y.shape[0] + np.sum(y))
    reci_neg = 2./(y.shape[0] - np.sum(y))
    K_y = torch.zeros_like(K_X_euclidean)
    K_y[y>0,y>0]=reci_pos
    K_y[y<0,y<0]=reci_neg

    if sample_weight is None:
        H = torch.eye(X.size(0), dtype=torch.float32, device=device) - torch.ones_like(K_X_euclidean)/X.size(0) 
    else:
        if np.any(sample_weight<0):
            raise ValueError('Negative weight detected!')
        sample_weight = torch.tensor(sample_weight, dtype=torch.float32, device=device)
        sample_weight /= sample_weight.sum()
        H = torch.diag(sample_weight) - torch.outer(sample_weight, sample_weight)
    HK_yH = H@K_y@H

    HSICs = torch.zeros(n_gammas, device=device)
    K_X = torch.zeros_like(K_X_euclidean)
    for it, gamma in enumerate(gammas):
        torch.exp(-gamma*K_X_euclidean, out=K_X)
        HSICs[it] = torch.sum(K_X*HK_yH)

    return gammas[torch.argmax(HSICs)].data.tolist()


class proxITR:
    def __init__(self, A,X,Z,W,Y, god=False, U=None, h0=None, q0=None, d_X_opt=None, learning_rate=0.1, n_epoch=10000, batch_size = 200, n_components = 150, opt='LBFGS', verbose=False, qtl = 0.4):
        self.A = A.reshape(-1)
        self.X = X
        self.Z = Z
        self.W = W
        self.Y = Y.reshape(-1)
        self.god=god
        if self.god:
            try:
                self.d_X_opt = d_X_opt.to_numpy(dtype=int).reshape(-1)
                self.U = U.to_numpy()
                self.h0 = h0.to_numpy()
                self.q0 = q0.to_numpy()
            except:
                pass
        # self.max_iter=10000
        # self.tol=1e-5
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.PipeNonLinear = pipeline.Pipeline([("Scaler", Scaler()),
                                  ("feature_map", 
                                        Nystroem(kernel="rbf", n_components=150, random_state=None)),
                                  ("svm",
                                        RegimeLearner(batch_size=self.batch_size, n_epoch=self.n_epoch))])
        self.PipeLinear = pipeline.Pipeline([("Scaler", Scaler()),
                                  ("svm", 
                                        RegimeLearner(batch_size=self.batch_size, n_epoch=self.n_epoch))])
        self.opt = opt # 'LBFGS' or 'SGD'
        self.n_components = n_components
        self.verbose = verbose
        self.qtl = qtl
        

    def fit_h0_a0(self, n_components=20, gamma_f='auto', gamma_h=0.1, alpha_scale='auto',index='all'):
        """
        RKHS_h0_a0.predict can be used to calculate h0(W,0,X)
        index is rows of data: if ='all', then use all data, otherwise is a 1 dim np.array
        """
        if _check_all(index): 
            index = (1-self.A).astype(bool)
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = (1-self.A).astype(bool) & fill

        RKHS_h0_a0 = ApproxRKHSIV(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h, alpha_scale=alpha_scale)\
                 .fit(np.concatenate((self.W,self.X),axis=1)[index,:],
                      self.Y[index],
                      np.concatenate((self.Z,self.X),axis=1)[index,:])
        return RKHS_h0_a0.predict
    def fit_h0_a0_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales = 'auto', n_components=20, cv=5, index='all'):
        """
        RKHS_h0_a0.predict can be used to calculate h0(W,0,X)
        index is rows of data: if ='all', then use all data, otherwise is a 1 dim np.array
        """
        if _check_all(index): 
            index = (1-self.A).astype(bool)
            index_all=True
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = (1-self.A).astype(bool) & fill
            index_all=False

        RKHS_h0_a0 = ApproxRKHSIVCV(gamma_f=gamma_f,n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, cv=cv, n_components=n_components).fit(
                      np.concatenate((self.W,self.X),axis=1)[index,:],
                      self.Y[index],
                      np.concatenate((self.Z,self.X),axis=1)[index,:])

        # save tuned parameters and predictor to self if index='all'
        if index_all: 
            self.gamma_f_h0_a0     = RKHS_h0_a0.gamma_f
            self.gamma_h_h0_a0     = RKHS_h0_a0.gamma_h
            self.alpha_scale_h0_a0 = RKHS_h0_a0.best_alpha_scale
            self.predict_h0_a0     = RKHS_h0_a0.predict
            self.cv_h0_a0          = cv

        return RKHS_h0_a0.gamma_f, RKHS_h0_a0.gamma_h, RKHS_h0_a0.best_alpha_scale, RKHS_h0_a0.predict

    def fit_h0_a1(self, n_components=20, gamma_f='auto', gamma_h=0.1, alpha_scale='auto',index='all'):
        """
        RKHS_h0_a1.predict can be used to calculate h0(W,1,X)
        index is rows of data: if ='all', then use all data, otherwise is a 1 dim np.array
        """
        if _check_all(index): 
            index = self.A.astype(bool)
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = self.A.astype(bool) & fill

        RKHS_h0_a1 = ApproxRKHSIV(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h, alpha_scale=alpha_scale)\
                 .fit(np.concatenate((self.W,self.X),axis=1)[index,:],
                      self.Y[index],
                      np.concatenate((self.Z,self.X),axis=1)[index,:])
        return RKHS_h0_a1.predict
    def fit_h0_a1_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales = 'auto', n_components=20, cv=5, index='all'):
        """
        RKHS_h0_a1.predict can be used to calculate h0(W,1,X)
        index is rows of data: if ='all', then use all data, otherwise is a 1 dim np.array
        """
        if _check_all(index): 
            index = self.A.astype(bool)
            index_all=True
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = self.A.astype(bool) & fill
            index_all=False

        RKHS_h0_a1 = ApproxRKHSIVCV(gamma_f=gamma_f,n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, cv=cv, n_components=n_components).fit(
                      np.concatenate((self.W,self.X),axis=1)[index,:],
                      self.Y[index],
                      np.concatenate((self.Z,self.X),axis=1)[index,:])

        # save tuned parameters and predictor to self if index='all'
        if index_all: 
            self.gamma_f_h0_a1     = RKHS_h0_a1.gamma_f
            self.gamma_h_h0_a1     = RKHS_h0_a1.gamma_h
            self.alpha_scale_h0_a1 = RKHS_h0_a1.best_alpha_scale
            self.predict_h0_a1     = RKHS_h0_a1.predict
            self.cv_h0_a1          = cv

        return RKHS_h0_a1.gamma_f, RKHS_h0_a1.gamma_h, RKHS_h0_a1.best_alpha_scale, RKHS_h0_a1.predict

    def predict_h0(self, n_components=20, gamma_f='auto', gamma_h=0.1):
        """
        Evaluate the performance of fit_h0_a0 and fit_h0_a1 using all data
        """
        self.h0_est = np.ndarray(self.A.shape[0])
        predict_h0_a0 = self.fit_h0_a0(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h)
        predict_h0_a1 = self.fit_h0_a1(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h)
        self.h0_est[(1-self.A).astype(bool)] = predict_h0_a0(np.concatenate((self.W,self.X),axis=1)[(1-self.A).astype(bool),:])
        self.h0_est[self.A.astype(bool)]     = predict_h0_a1(np.concatenate((self.W,self.X),axis=1)[self.A.astype(bool),:])
        #self.RKHS_h0.predict(np.concatenate((self.W.reshape(-1,1),self.A.reshape(-1,1),self.X),axis=1))
        if self.god:
            print("RMSE of h0_hat: ",sklearn.metrics.mean_squared_error(self.h0, self.h0_est, squared=False), '\n')
        return
    def predict_h0_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales, n_components, cv):
        """
        Evaluate the performance of fit_h0_a0 and fit_h0_a1 using all data
        """
        self.h0_est = np.ndarray(self.A.shape[0])
        _,_,_,predict_h0_a0 = self.fit_h0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv, index='all')
        _,_,_,predict_h0_a1 = self.fit_h0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv, index='all')
        return predict_h0_a0, predict_h0_a1


    def fit_q0_a0(self, n_components=20, gamma_f='auto', gamma_h=0.1, alpha_scale='auto',index='all'):
        """
        RKHS_q0_a0.predict can be used to calculate q(Z,0,X) =>q0(Z,X)
        """
        if _check_all(index): 
            index = (1-self.A).astype(bool)
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = (1-self.A).astype(bool) & fill

        RKHS_q0_a0 = ApproxRKHSIV_q(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h, alpha_scale=alpha_scale).fit(
                      np.concatenate((self.Z, self.X),axis=1), 
                      np.ones_like(index), 
                      np.concatenate((self.X, self.W),axis=1),
                      index)
        
        return RKHS_q0_a0.predict
    def fit_q0_a0_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales='auto', n_components=20, cv=5, index='all'):
        """
        RKHS_q0_a0.predict can be used to calculate q(Z,0,X) =>q0(Z,X)
        """
        if _check_all(index): 
            index = (1-self.A).astype(bool)
            index_all=True
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = (1-self.A).astype(bool) & fill
            index_all=False

        RKHS_q0_a0 = ApproxRKHSIVCV_q(gamma_f=gamma_f,n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, cv=cv, n_components=n_components).fit(
                      np.concatenate((self.Z, self.X),axis=1), 
                      np.ones_like(index), 
                      np.concatenate((self.X, self.W),axis=1), 
                      index)

        # save tuned parameters and predictor to self if index='all'
        if index_all: 
            self.gamma_f_q0_a0     = RKHS_q0_a0.gamma_f
            self.gamma_h_q0_a0     = RKHS_q0_a0.gamma_h
            self.alpha_scale_q0_a0 = RKHS_q0_a0.best_alpha_scale
            self.predict_q0_a0     = RKHS_q0_a0.predict
            self.cv_q0_a0          = cv
        
        return RKHS_q0_a0.gamma_f, RKHS_q0_a0.gamma_h, RKHS_q0_a0.best_alpha_scale, RKHS_q0_a0.predict

    def fit_q0_a1(self, n_components=20, gamma_f='auto', gamma_h=0.1, alpha_scale='auto',index='all'):
        """
        RKHS_q0_a1.predict can be used to calculate q(Z,1,X) =>q1(Z,X)
        """
        if _check_all(index): 
            index = self.A.astype(bool)
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = self.A.astype(bool) & fill

        RKHS_q0_a1 = ApproxRKHSIV_q(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h, alpha_scale=alpha_scale).fit(
                      np.concatenate((self.Z, self.X),axis=1), 
                      np.ones_like(index), 
                      np.concatenate((self.X, self.W),axis=1), 
                      index)

        return RKHS_q0_a1.predict 
    def fit_q0_a1_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales='auto', n_components=20, cv=5, index='all'):
        """
        RKHS_q0_a1.predict can be used to calculate q(Z,1,X) =>q1(Z,X)
        """
        if _check_all(index): 
            index = self.A.astype(bool)
            index_all = True
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = self.A.astype(bool) & fill
            index_all = False

        RKHS_q0_a1 = ApproxRKHSIVCV_q(gamma_f=gamma_f,n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, cv=cv, n_components=n_components).fit(
                      np.concatenate((self.Z, self.X),axis=1), 
                      np.ones_like(index), 
                      np.concatenate((self.X, self.W),axis=1), 
                      index)

        # save tuned parameters and predictor to self if index='all'
        if index_all: 
            self.gamma_f_q0_a1     = RKHS_q0_a1.gamma_f
            self.gamma_h_q0_a1     = RKHS_q0_a1.gamma_h
            self.alpha_scale_q0_a1 = RKHS_q0_a1.best_alpha_scale
            self.predict_q0_a1     = RKHS_q0_a1.predict
            self.cv_q0_a1          = cv

        return RKHS_q0_a1.gamma_f, RKHS_q0_a1.gamma_h, RKHS_q0_a1.best_alpha_scale, RKHS_q0_a1.predict

    def predict_q0(self, n_components, index):
        """
        q0_est : calculate estimate of q0(Z_i, A_i, L_i) on data[index,:]
        """
        
        self.gamma_f_q0_a0 = self.gamma_f_q0_a1 = 10
        self.gamma_h_q0_a0 = self.gamma_h_q0_a1= 5
        self.alpha_scale_q0_a0 = self.alpha_scale_q0_a1 = 1
        
        q0_est = np.ndarray(self.A.shape[0])
        predict_q0_a0 = self.fit_q0_a0(n_components=n_components, gamma_f=self.gamma_f_q0_a0, gamma_h=self.gamma_h_q0_a0, alpha_scale = self.alpha_scale_q0_a0, index = index)
        predict_q0_a1 = self.fit_q0_a1(n_components=n_components, gamma_f=self.gamma_f_q0_a1, gamma_h=self.gamma_h_q0_a1, alpha_scale = self.alpha_scale_q0_a1, index = index)
        q0_est[(1-self.A).astype(bool)] = predict_q0_a0(np.concatenate((self.Z, self.X),axis=1)[(1-self.A).astype(bool),:])
        q0_est[self.A.astype(bool)]     = predict_q0_a1(np.concatenate((self.Z, self.X),axis=1)[self.A.astype(bool),:])

        return predict_q0_a0, predict_q0_a1
    
    def predict_q0_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales='auto', n_components=20, cv=5):
        """
        self.q0_est : calculate estimate of q0(Z_i, A_i, L_i) based on all data
        """
        self.q0_est = np.ndarray(self.A.shape[0])
        _,_,_,predict_q0_a0 = self.fit_q0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, n_components=n_components, cv=cv)
        _,_,_,predict_q0_a1 = self.fit_q0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, n_components=n_components, cv=cv)
        return predict_q0_a0, predict_q0_a1
    