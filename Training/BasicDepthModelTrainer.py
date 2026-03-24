import torch
import os

from Training.Trainer import Trainer, TrainerInitializationData, CallbackIntervalType, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARN_RATE
from Datasets.DepthPredictionDataset import DepthPredictionDataset
from Models.BasicDepthPredictionModel import DepthPredictionModel
from Utilities.LossFunctions import ScaleInvariantLoss, EdgeAwareSmoothnessLoss, EdgeFocusedScaleInvariantLoss
from Utilities.WandBPerformanceTracker import WeightsAndBiasesLogger

DATASET_SCALE_FACTORS = {
        "KITTI": 256.0,
        "NYU": 1.0,
        "MegaDepth": 1.0
    }

class BasicDepthModelTrainer(Trainer):
    def __init__(self, datasetName:str, device:torch.device, runName:str, checkpointDirectory:str, validationSetRatio:float=0.1, batchSize:int=DEFAULT_BATCH_SIZE, epochs:int=DEFAULT_EPOCHS, learnRate:float=DEFAULT_LEARN_RATE) -> None:
        initializationData = TrainerInitializationData()

        scaleFactor = DATASET_SCALE_FACTORS[datasetName]

        trainingDataset = DepthPredictionDataset(device, datasetName, scaleFactor=scaleFactor)
        initializationData.trainingDataset = trainingDataset

        self.checkpointDirectory = os.path.join(checkpointDirectory, runName)
        self.runName = runName

        validationSetSize = int(len(trainingDataset) * validationSetRatio)

        validationDatasetIndices = trainingDataset.PartitionValidationSet(validationSetSize)
        validationDataset = DepthPredictionDataset(device, datasetName, validationDatasetIndices, scaleFactor=scaleFactor)
        initializationData.validationDataset = validationDataset

        model = DepthPredictionModel().to(device)
        #model = UNetDepthPredictionModel().to(device)
        initializationData.model = model

        initializationData.batchSize = batchSize
        initializationData.epochs = epochs
        initializationData.learnRate = learnRate

        initializationData.device = device

        initializationData.lossFunction = self.LossFunction
        
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
        self.AddCallback(self.SaveModelCheckpoint, intervalType=CallbackIntervalType.EVERY_N_EPOCHS, interval=1)

    def LossFunction(self, inputTensor:torch.Tensor, outputTensor:torch.Tensor, groundTruthTensor:torch.Tensor, validMask:torch.Tensor) -> torch.Tensor:
        mainLoss = ScaleInvariantLoss(inputTensor, outputTensor, groundTruthTensor, validMask)

        smoothnessLoss = EdgeAwareSmoothnessLoss(inputTensor, outputTensor, groundTruthTensor, validMask)

        edgeFocusedLoss = EdgeFocusedScaleInvariantLoss(inputTensor, outputTensor, groundTruthTensor, validMask)

        totalLoss = (1.5 * mainLoss) + (1.0 * smoothnessLoss) + (1.0 * edgeFocusedLoss)

        return totalLoss

    def TrainingStep(self, inputData:tuple) -> torch.Tensor:
        inputTensor, gtTensor, validMask = inputData

        outputTensor = self.model(inputTensor)

        loss:torch.Tensor = self.lossFunction(inputTensor, outputTensor, gtTensor, validMask)

        if self.currentTrainingBatchIndex % self.lossLogInterval == 0:
            self.logger.LogData({
                "trainingLoss": loss.detach().item()
            })

        if self.currentTrainingBatchIndex % self.imageLogInterval == 0:
            self.logger.LogImage([inputTensor, gtTensor, validMask, outputTensor], [False, False, False, True], self.currentEpoch, self.currentTrainingBatchIndex)

        self.logger.NextStep()

        return loss

    def ValidationStep(self, validationBatchIndex:int, inputData:tuple) -> torch.Tensor:
        inputTensor, gtTensor, validMask = inputData

        outputTensor = self.model(inputTensor)

        loss = self.lossFunction(inputTensor, outputTensor, gtTensor, validMask)

        return loss
    
    def LogEpochLoss(self) -> None:
        self.logger.LogData({
            "averageTrainingLoss": self.currentAverageTrainingLoss.detach().item(),
            "averageValidationLoss": self.currentAverageValidationLoss.detach().item()
        })

    def SaveModelCheckpoint(self) -> None:
        onnxModelPath = os.path.join(self.checkpointDirectory, "onnx")
        os.makedirs(onnxModelPath, exist_ok=True)

        checkpointPath = os.path.join(self.checkpointDirectory, "checkpoints")
        os.makedirs(checkpointPath, exist_ok=True)

        onnxFileName = f"{self.runName}_e{self.currentEpoch}_i{self.currentTrainingBatchIndex}.onnx"
        onnxFilePath = os.path.join(onnxModelPath, onnxFileName)

        checkpointFileName = f"{self.runName}_e{self.currentEpoch}_i{self.currentTrainingBatchIndex}.ckpt"
        checkpointFilePath = os.path.join(checkpointPath, checkpointFileName)

        isModelTraining = self.model.training
        if isModelTraining:
            self.model.eval()

        exampleInput = (torch.zeros((1, 3, 256, 256)).to(self.device),)
        onnxProgram = torch.onnx.export(self.model, exampleInput, dynamo=True, 
                                     input_names=["input"],
                                     output_names=["output"],
                                     dynamic_axes={
                                         "input": {2: "height", 3: "width"}
                                     })
        
        onnxProgram.save(onnxFilePath)

        torch.save(self.model.state_dict(), checkpointFilePath)

        if isModelTraining:
            self.model.train()