from torch.utils.data import Dataset
import torch
import torchvision.io

import os
import random

class DepthPredictionDataset(Dataset):
    def __init__(self, device:torch.device, datasetName:str, indices:list[int]=None, scaleFactor:float=1.0, maximumImageDimension:int=768) -> None:
        super().__init__()

        random.seed(12345)

        self.maximumDimension = maximumImageDimension

        self.device = device
        self.datasetName = datasetName

        self.directoryPath = os.path.join("data", datasetName)

        self.depthPath = os.path.join(self.directoryPath, "depth")
        self.rgbPath = os.path.join(self.directoryPath, "rgb")

        self.scaleFactor = scaleFactor

        self.depthFiles = os.listdir(self.depthPath)
        self.depthFiles.sort()
        self.rgbFiles = os.listdir(self.rgbPath)
        self.rgbFiles.sort()

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

    def CropTensorToMaximumSize(self, inputTensor:torch.Tensor) -> torch.Tensor:
        C, H, W = inputTensor.shape

        # Determine final crop size (don’t exceed original)
        cropH = min(H, self.maximumDimension)
        cropW = min(W, self.maximumDimension)

        if cropH % 16 != 0:
            cropH = (cropH // 16) * 16

        if cropW % 16 != 0:
            cropW = (cropW // 16) * 16

        # Compute top-left corner for centered crop
        top = (H - cropH) // 2
        left = (W - cropW) // 2

        # Crop
        return inputTensor[:, top:top + cropH, left:left + cropW]

    def CreateDataTensor(self, filePath:str, readMode:torchvision.io.ImageReadMode, scaleFactor:float=1.0) -> torch.Tensor:
        resultTensor = torchvision.io.read_image(filePath, readMode).to(self.device).float()

        resultTensor = self.CropTensorToMaximumSize(resultTensor)

        resultTensor /= scaleFactor

        return resultTensor

    def PartitionValidationSet(self, validationSetSize:int) -> list[int]:
        result = self.indices[:validationSetSize]

        self.indices = self.indices[validationSetSize:]

        return result