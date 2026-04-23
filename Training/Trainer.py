from enum import Enum
import torch
from tqdm import tqdm
import gc

DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARN_RATE = 0.0001
DEFAULT_EPOCHS = 100

class CallbackIntervalType(Enum):
    EVERY_N_EPOCHS = 0
    EVERY_N_ITERATIONS = 1

class TrainerInitializationData():
    def __init__(self) -> None:
        self.trainingDataset = None
        self.validationDataset = None
 
        self.device = None

        self.model = None

        self.batchSize = DEFAULT_BATCH_SIZE
        self.learnRate = DEFAULT_LEARN_RATE
        self.epochs = DEFAULT_EPOCHS

        self.optimizer = None
        self.lrScheduler = None
        self.lossFunction = None

class Trainer():
    def __init__(self, initializationData:TrainerInitializationData) -> None:
        self.trainingDataset:torch.utils.data.Dataset = initializationData.trainingDataset
        self.validationDataset:torch.utils.data.Dataset = initializationData.validationDataset

        self.device:torch.device = initializationData.device

        self.model:torch.nn.Module = initializationData.model

        self.batchSize:int = initializationData.batchSize
        self.learnRate:float = initializationData.learnRate
        self.epochs:int = initializationData.epochs

        self.optimizer:torch.optim.Optimizer = initializationData.optimizer
        self.lrScheduler:torch.optim.lr_scheduler.LRScheduler = initializationData.lrScheduler
        self.lossFunction = initializationData.lossFunction

        self.validationIntervalType:CallbackIntervalType = CallbackIntervalType.EVERY_N_EPOCHS
        self.validationInterval:int = 1

        self.currentAverageValidationLoss = 0.0
        self.currentAverageTrainingLoss = 0.0
        self.currentTrainingLoss = 0.0

        self.currentEpoch = 0
        self.currentTrainingBatchIndex = 0

        self.enableDebugLogging = True

        self.callbacks:list[dict] = []

        self.CreateDataLoaders()

    def CreateDataLoaders(self) -> None:
        self.trainingLoader = torch.utils.data.DataLoader(self.trainingDataset, batch_size=self.batchSize, shuffle=True)
        self.validationLoader = torch.utils.data.DataLoader(self.validationDataset, batch_size=self.batchSize, shuffle=False)

    def SetValidationInterval(self, intervalType:CallbackIntervalType, interval:int) -> None:
        self.validationIntervalType = intervalType
        self.validationInterval = interval

    def AddCallback(self, callbackFunction, intervalType:CallbackIntervalType, interval:int) -> None:
        newCallbackObject = {
            "function": callbackFunction,
            "intervalType": intervalType,
            "interval": interval
        }

        self.callbacks.append(newCallbackObject)

    def TrainingLoop(self) -> None:
        trainBatchCount = len(self.trainingLoader)

        #loop through epochs
        for epochIndex in range(self.epochs):
            self.currentEpoch = epochIndex

            #set model to train mode
            self.model.train()

            if self.enableDebugLogging:
                print(f"Starting epoch {epochIndex}")

            #loss = 0 for recording per-epoch loss
            trainingLoss = 0.0
            #loop through training batches
            for batchIndex, inputData in tqdm(enumerate(self.trainingLoader), total=trainBatchCount):
                self.currentTrainingBatchIndex = batchIndex + (trainBatchCount * epochIndex)

                self.optimizer.zero_grad()
                
                currentLoss:torch.Tensor = self.TrainingStep(inputData)

                #apply loss
                currentLoss.backward()
                self.optimizer.step()

                self.currentTrainingLoss = currentLoss.detach().item()

                #step scheduler if needed
                self.lrScheduler.step(epochIndex + (batchIndex / trainBatchCount))

                trainingLoss += self.currentTrainingLoss

                if self.validationIntervalType == CallbackIntervalType.EVERY_N_ITERATIONS and (self.currentTrainingBatchIndex + 1) % self.validationInterval == 0:
                    self.currentAverageValidationLoss = self.ValidationLoop()

                for callbackObject in self.callbacks:
                    if callbackObject["intervalType"] == CallbackIntervalType.EVERY_N_ITERATIONS and (self.currentTrainingBatchIndex + 1) % callbackObject["interval"] == 0:
                        callbackObject["function"]()

                gc.collect()
                torch.cuda.empty_cache()

            self.currentAverageTrainingLoss = trainingLoss / trainBatchCount

            if self.enableDebugLogging:
                print(f"Epoch {epochIndex}:\n\tTraining Loss: {self.currentAverageTrainingLoss}\n")

            if self.validationIntervalType == CallbackIntervalType.EVERY_N_EPOCHS and (epochIndex + 1) % self.validationInterval == 0:
                self.currentAverageValidationLoss = self.ValidationLoop()

            for callbackObject in self.callbacks:
                if callbackObject["intervalType"] == CallbackIntervalType.EVERY_N_EPOCHS and (epochIndex + 1) % callbackObject["interval"] == 0:
                    callbackObject["function"]()

    def ValidationLoop(self) -> float:
        validationBatchCount = len(self.validationLoader)
        
        torch.set_grad_enabled(False)

        #set model to eval mode
        self.model.eval()

        #validation loss = 0 for recording
        validationLoss = 0.0
        #loop through validation batches
        for batchIndex, inputData in tqdm(enumerate(self.validationLoader), total=validationBatchCount):
            loss:torch.Tensor = self.ValidationStep(batchIndex, inputData)

            #add to total
            validationLoss += loss.detach().item()

        #report epoch index and train/validation loss
        averageValidationLoss = validationLoss / validationBatchCount

        if self.enableDebugLogging:
            print(f"Epoch {self.currentEpoch}, Iteration {self.currentTrainingBatchIndex}:\n\tValidation Loss: {averageValidationLoss}\n")

        torch.set_grad_enabled(True)

        return averageValidationLoss

    def TrainingStep(self, inputData:tuple) -> torch.Tensor:
        return torch.tensor(0).to(self.device)

    def ValidationStep(self, validationBatchIndex:int, inputData:tuple) -> torch.Tensor:
        return torch.tensor(0).to(self.device)