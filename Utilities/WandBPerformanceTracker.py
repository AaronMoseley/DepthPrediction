import wandb
import torch
import numpy as np

from Utilities.PerformanceLogger import PerformanceLogger

class WeightsAndBiasesLogger(PerformanceLogger):
    def __init__(self, projectName:str, runName:str, configData:dict) -> None:
        self.projectName = projectName
        self.runName = runName
        
        runConfig = configData.copy()

        if len(runName) > 0:
            runConfig["runName"] = runName
        
        self.run = wandb.init(
            entity="amoseley018",
            project=projectName,
            config=runConfig,
            name=runName
        )

        self.storedData = {}

    def LogData(self, data:dict, step:int=None) -> None:
        if step is not None:
            self.run.log(data, step=step)
            return
        
        for key in data:
            self.storedData[key] = data[key]

    def NextStep(self) -> None:
        if len(self.storedData) == 0:
            return
        
        self.run.log(self.storedData)

        self.storedData = {}

    def LogImage(self, tensorList:list[torch.Tensor], isDepth:list[bool], epochIndex:int, batchIndex:int) -> None:
        widthSum = 0
        maxHeight = 0
        batchSize = 0
        
        #convert image tensor to numpy array
        arrayList:list[np.ndarray] = []
        for i, tensor in enumerate(tensorList):
            newArray = tensor.detach().cpu().numpy()

            if isDepth[i] == True:
                newArray -= np.min(newArray)
                newArray = np.exp(newArray)
                newArray /= np.max(newArray)
                newArray *= 255.0
            else:
                newArray /= np.max(newArray)
                newArray *= 255.0

            newArray = np.asarray(newArray, dtype=np.uint8)

            currentBatchSize, channels, currentHeight, currentWidth = newArray.shape

            widthSum += currentWidth
            maxHeight = max(maxHeight, currentHeight)
            batchSize = max(batchSize, currentBatchSize)

            if channels == 1:
                newArray = np.tile(newArray, (1, 3, 1, 1))

            arrayList.append(newArray)

        sideBySideArray = np.zeros((batchSize, 3, maxHeight, widthSum), dtype=np.uint8)

        startingPosition = 0
        for imageArray in arrayList:
            endPosition = startingPosition + imageArray.shape[3]
            arrayHeight = imageArray.shape[2]

            sideBySideArray[:, :, :arrayHeight, startingPosition:endPosition] = imageArray

            startingPosition = endPosition

        #create new empty array, stack the arrays vertically
        stackedArray = np.zeros((3, maxHeight * batchSize, widthSum), dtype=np.uint8)
        for i in range(batchSize):
            stackedArray[:, i * maxHeight:(i + 1) * maxHeight, :] = sideBySideArray[i, :, :, :]

        stackedArray = np.transpose(stackedArray, axes=(1, 2, 0))

        #log the image
        finalImage = wandb.Image(stackedArray, caption=f"Epoch: {epochIndex}, Batch: {batchIndex}")
        self.run.log({
            "images": [finalImage]
        })

    def FinishRun(self) -> None:
        self.run.finish()