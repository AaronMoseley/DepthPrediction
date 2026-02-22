import sys
from Utilities.ArgumentManager import ArgumentManager
import torch
from torch.utils.data import Dataset

from Datasets.DepthPredictionDataset import DepthPredictionDataset

def TrainModel(model:torch.nn.Module, lossFunction, trainingDataset:Dataset, validationDataset:Dataset) -> None:
    pass

if __name__ == "__main__":
    print("training model")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    argManager = ArgumentManager(sys.argv)

    trainingDataset = DepthPredictionDataset(device, argManager.dataset)

    validationSetSize = int(len(trainingDataset) * 0.1)
    if "validationSetRatio" in argManager:
        validationSetSize = int(len(trainingDataset) * argManager.validationSetRatio)

    validationDatasetIndices = trainingDataset.PartitionValidationSet(validationSetSize)
    validationDataset = DepthPredictionDataset(device, argManager.dataset, validationDatasetIndices)