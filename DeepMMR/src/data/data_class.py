from typing import NamedTuple, Optional
import numpy as np
import torch
from sklearn.model_selection import train_test_split

'''
Data classes for MMR datasets
'''

class MMRTrainDataSet_h(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: np.ndarray

class MMRTestDataSet_h(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: np.ndarray

class MMRTrainDataSet_q(NamedTuple):
    treatment: np.ndarray
    treatment_target:np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: np.ndarray


class MMRTestDataSet_q(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: np.ndarray
    structural: np.ndarray


class MMRTrainDataSetTorch_h(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: MMRTrainDataSet_h):
        backdoor = None
        if train_data.backdoor is not None:
            backdoor = torch.tensor(train_data.backdoor, dtype=torch.float32)
        return MMRTrainDataSetTorch_h(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(train_data.treatment_proxy, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(train_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=backdoor,
                                   outcome=torch.tensor(train_data.outcome, dtype=torch.float32))

    def to_gpu(self):
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor.cuda()
        return MMRTrainDataSetTorch_h(treatment=self.treatment.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=backdoor,
                                   outcome=self.outcome.cuda())

class MMRTrainDataSetTorch_q(NamedTuple):
    treatment: torch.Tensor
    treatment_target:torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: MMRTrainDataSet_q):
        if hasattr(train_data, 'treatment_target'):
            A_target = train_data.treatment_target
        else:
            A_target = np.zeros(train_data.treatment.shape)
        return MMRTrainDataSetTorch_q(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                   treatment_target=torch.tensor(A_target, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(train_data.treatment_proxy, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(train_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=torch.tensor(train_data.backdoor, dtype=torch.float32),
                                   outcome=torch.tensor(train_data.outcome, dtype=torch.float32))

    def to_gpu(self):
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor.cuda()
        return MMRTrainDataSetTorch_q(treatment=self.treatment.cuda(),
                                   treatment_target=self.treatment_target.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=backdoor,
                                   outcome=self.outcome.cuda())


class MMRTestDataSetTorch_h(NamedTuple):
    treatment: torch.Tensor
    outcome_proxy: torch.Tensor
    backdoor: torch.Tensor
    structural: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, test_data: MMRTestDataSet_h):
        structural = None
        if hasattr(test_data, 'structural'):
            structural = torch.tensor(test_data.structural, dtype=torch.float32)
        return MMRTestDataSetTorch_h(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(test_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=torch.tensor(test_data.backdoor, dtype=torch.float32),
                                   structural=structural)

    def to_gpu(self):
        structural = None
        if self.structural is not None:
            structural = self.structural.cuda()
        return MMRTestDataSetTorch_h(treatment=self.treatment.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=self.backdoor.cuda(),
                                   structural=structural)

class MMRTestDataSetTorch_q(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: torch.Tensor
    structural: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, test_data: MMRTestDataSet_q):
        structural = None
        if hasattr(test_data, 'structural'):
            structural = torch.tensor(test_data.structural, dtype=torch.float32)
        return MMRTestDataSetTorch_q(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(test_data.treatment_proxy, dtype=torch.float32),
                                   outcome=torch.tensor(test_data.outcome, dtype=torch.float32),
                                   backdoor=torch.tensor(test_data.backdoor, dtype=torch.float32),
                                   structural=structural)

    def to_gpu(self):
        structural = None
        if self.structural is not None:
            structural = self.structural.cuda()
        return MMRTestDataSetTorch_q(treatment=self.treatment.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   outcome=self.outcome.cuda(),
                                   backdoor=self.backdoor.cuda(),
                                   structural=structural)


def split_train_data(train_data: MMRTrainDataSet_h, split_ratio=0.5):
    if split_ratio < 0.0:
        return train_data, train_data

    n_data = train_data[0].shape[0]
    idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=split_ratio)

    def get_data(data, idx):
        return data[idx] if data is not None else None

    train_1st_data = MMRTrainDataSet_h(*[get_data(data, idx_train_1st) for data in train_data])
    train_2nd_data = MMRTrainDataSet_h(*[get_data(data, idx_train_2nd) for data in train_data])
    return train_1st_data, train_2nd_data
