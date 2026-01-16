import sys
from Utilities.ArgumentManager import ArgumentManager
import torch
from torch.utils.data import Dataset

def TrainModel(model:torch.nn.Module, lossFunction:function, trainingDataset:Dataset, validationDataset:Dataset) -> None:
    pass

if __name__ == "__main__":
    print("training model")
    argManager = ArgumentManager(sys.argv)