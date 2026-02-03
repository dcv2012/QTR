import pathlib
import os.path as op
import numpy as np
import pandas as pd

from src.data.data_class import MMRTrainDataSet_h, MMRTestDataSet_h

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("data/right_heart_catheterization")


def generate_train_rhc(use_all_X: bool) -> MMRTrainDataSet_h:
    # load train split
    train_split = pd.read_csv(op.join(DATA_PATH, "rhc_train.csv"))

    # load X feature list
    if use_all_X:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_allfeatures_list.csv"))
    else:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_significantfeatures_list.csv"))
        # X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_selectedfeatures.csv"))
        
    # subset the train split to X_names
    train_X = train_split[X_names.variable.tolist()].to_numpy()

    # separate the key proximal variables
    rhc_treatment = np.expand_dims(train_split['swang1'].to_numpy(), axis=-1)
    survival_time = np.expand_dims(train_split['t3d30'].to_numpy(), axis=-1)
    Z_pafi = np.expand_dims(train_split['pafi1'].to_numpy(), axis=-1)
    Z_paco21 = np.expand_dims(train_split['paco21'].to_numpy(), axis=-1)
    W_ph1 = np.expand_dims(train_split['ph1'].to_numpy(), axis=-1)
    W_hema1 = np.expand_dims(train_split['hema1'].to_numpy(), axis=-1)

    return MMRTrainDataSet_h(treatment=rhc_treatment,
                          treatment_proxy=np.c_[Z_pafi, Z_paco21],
                          outcome_proxy=np.c_[W_ph1, W_hema1],
                          outcome=survival_time,
                          backdoor=train_X)


def generate_val_rhc(use_all_X: bool) -> MMRTrainDataSet_h:
    # load the validation split
    val_split = pd.read_csv(op.join(DATA_PATH, "rhc_val.csv"))

    # load X feature list
    if use_all_X:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_allfeatures_list.csv"))
    else:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_significantfeatures_list.csv"))
        # X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_selectedfeatures.csv"))
    
    # subset the validation split to X_names
    val_X = val_split[X_names.variable.tolist()].to_numpy()

    # separate the key proximal variables
    rhc_treatment = np.expand_dims(val_split['swang1'].to_numpy(), axis=-1)
    survival_time = np.expand_dims(val_split['t3d30'].to_numpy(), axis=-1)
    Z_pafi = np.expand_dims(val_split['pafi1'].to_numpy(), axis=-1)
    Z_paco21 = np.expand_dims(val_split['paco21'].to_numpy(), axis=-1)
    W_ph1 = np.expand_dims(val_split['ph1'].to_numpy(), axis=-1)
    W_hema1 = np.expand_dims(val_split['hema1'].to_numpy(), axis=-1)

    return MMRTrainDataSet_h(treatment=rhc_treatment,
                          treatment_proxy=np.c_[Z_pafi, Z_paco21],
                          outcome_proxy=np.c_[W_ph1, W_hema1],
                          outcome=survival_time,
                          backdoor=val_X)



def generate_test_rhc(use_all_X: bool) -> MMRTestDataSet_h:

    # load the test split
    test_split = pd.read_csv(op.join(DATA_PATH, "rhc_test.csv"))

    # load X feature list
    if use_all_X:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_allfeatures_list.csv"))
    else:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_significantfeatures_list.csv"))
        # X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_selectedfeatures.csv"))
        
    # subset the test split to X_names
    test_X = test_split[X_names.variable.tolist()].to_numpy()
    
    rhc_treatment = np.expand_dims(test_split['swang1'].to_numpy(), axis=-1)
    survival_time = np.expand_dims(test_split['t3d30'].to_numpy(), axis=-1)
    Z_pafi = np.expand_dims(test_split['pafi1'].to_numpy(), axis=-1)
    Z_paco21 = np.expand_dims(test_split['paco21'].to_numpy(), axis=-1)
    W_ph1 = np.expand_dims(test_split['ph1'].to_numpy(), axis=-1)
    W_hema1 = np.expand_dims(test_split['hema1'].to_numpy(), axis=-1)

    return MMRTrainDataSet_h(treatment=rhc_treatment,
                          treatment_proxy=np.c_[Z_pafi, Z_paco21],
                          outcome_proxy=np.c_[W_ph1, W_hema1],
                          outcome=survival_time,
                          backdoor=test_X)
