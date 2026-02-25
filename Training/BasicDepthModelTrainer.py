import torch

from Training.Trainer import Trainer, TrainerInitializationData, CallbackIntervalType, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARN_RATE
from Datasets.DepthPredictionDataset import DepthPredictionDataset
from Models.BasicDepthPredictionModel import DepthPredictionModel
from Utilities.LossFunctions import ScaleInvariantLoss
from Utilities.WandBPerformanceTracker import WeightsAndBiasesLogger

DATASET_SCALE_FACTORS = {
        "KITTI": 256.0,
        "NYU": 1.0,
        "MegaDepth": 1.0
    }

class BasicDepthModelTrainer(Trainer):
    def __init__(self, datasetName:str, device:torch.device, runName:str, validationSetRatio:float=0.1, batchSize:int=DEFAULT_BATCH_SIZE, epochs:int=DEFAULT_EPOCHS, learnRate:float=DEFAULT_LEARN_RATE) -> None:
        initializationData = TrainerInitializationData()

        scaleFactor = DATASET_SCALE_FACTORS[datasetName]

        trainingDataset = DepthPredictionDataset(device, datasetName, scaleFactor=scaleFactor)
        initializationData.trainingDataset = trainingDataset

        validationSetSize = int(len(trainingDataset) * validationSetRatio)

        validationDatasetIndices = trainingDataset.PartitionValidationSet(validationSetSize)
        validationDataset = DepthPredictionDataset(device, datasetName, validationDatasetIndices, scaleFactor=scaleFactor)
        initializationData.validationDataset = validationDataset

        model = DepthPredictionModel().to(device)
        initializationData.model = model

        initializationData.batchSize = batchSize
        initializationData.epochs = epochs
        initializationData.learnRate = learnRate

        initializationData.device = device

        initializationData.lossFunction = ScaleInvariantLoss
        
        initializationData.optimizer = torch.optim.RAdam(model.parameters(), learnRate)
        initializationData.lrScheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(initializationData.optimizer, 10)

        self.lossLogInterval = 10
        self.imageLogInterval = 100

        super().__init__(initializationData)

        self.logger = WeightsAndBiasesLogger("Depth Prediction", runName, {
            "batchSize": self.batchSize,
            "learnRate": self.learnRate,
            "epochs": self.epochs,
            "dataset": datasetName
        })

        self.AddCallback(self.LogEpochLoss, intervalType=CallbackIntervalType.EVERY_N_EPOCHS, interval=1)

    def TrainingStep(self, inputData:tuple) -> torch.Tensor:
        inputTensor, gtTensor, validMask = inputData

        outputTensor = self.model(inputTensor)

        loss:torch.Tensor = self.lossFunction(outputTensor, gtTensor, validMask)

        if self.currentTrainingBatchIndex % self.lossLogInterval == 0:
            self.logger.LogData({
                "trainingLoss": loss.detach().item()
            }, step=self.currentTrainingBatchIndex)

        if self.currentTrainingBatchIndex % self.imageLogInterval == 0:
            self.logger.LogImage([inputTensor, gtTensor, validMask, outputTensor], self.currentEpoch, self.currentTrainingBatchIndex)

        return loss

    def ValidationStep(self, validationBatchIndex:int, inputData:tuple) -> torch.Tensor:
        inputTensor, gtTensor, validMask = inputData

        outputTensor = self.model(inputTensor)

        loss = self.lossFunction(outputTensor, gtTensor, validMask)

        return loss
    
    def LogEpochLoss(self) -> None:
        self.logger.LogData({
            "averageTrainingLoss": self.currentAverageTrainingLoss.detach().item(),
            "averageValidationLoss": self.currentAverageValidationLoss.detach().item()
        }, step=self.currentEpoch)