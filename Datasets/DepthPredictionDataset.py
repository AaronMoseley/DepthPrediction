from torch.utils.data import Dataset
import torch
import torchvision.io

import os
import random

class DepthPredictionDataset(Dataset):
    def __init__(self, device:torch.device, datasetName:str, indices:list[int]=None, scaleFactor:float=1.0) -> None:
        super().__init__()

        random.seed(12345)

        self.device = device
        self.datasetName = datasetName

        self.directoryPath = os.path.join("data", datasetName)

        self.depthPath = os.path.join(self.directoryPath, "depth")
        self.rgbPath = os.path.join(self.directoryPath, "rgb")

        self.scaleFactor = scaleFactor

        self.depthFiles = os.listdir(self.depthPath)
        self.rgbFiles = os.listdir(self.rgbPath)

        if indices is not None:
            self.indices = indices
            return

        self.indices = [i for i in range(len(self.rgbFiles))]
        random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index:int) -> tuple[torch.Tensor, torch.Tensor]:
        rgbFileName = self.rgbFiles[self.indices[index]]
        rgbTensor = self.CreateDataTensor(os.path.join(self.rgbPath, rgbFileName), torchvision.io.ImageReadMode.RGB, 255.0)

        depthFileName = self.depthFiles[self.indices[index]]
        depthTensor = self.CreateDataTensor(os.path.join(self.depthPath, depthFileName), torchvision.io.ImageReadMode.UNCHANGED, self.scaleFactor)

        validDataMask = torch.zeros_like(depthTensor)
        validDataMask[depthTensor >= 0.01] = 1

        return rgbTensor, depthTensor, validDataMask

    def CreateDataTensor(self, filePath:str, readMode:torchvision.io.ImageReadMode, scaleFactor:float=1.0) -> torch.Tensor:
        resultTensor = torchvision.io.read_image(filePath, readMode).to(self.device).float()

        resultTensor /= scaleFactor

        return resultTensor

    def PartitionValidationSet(self, validationSetSize:int) -> list[int]:
        result = self.indices[:validationSetSize]

        self.indices = self.indices[validationSetSize:]

        return result