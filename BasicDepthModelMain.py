import sys
from Utilities.ArgumentManager import ArgumentManager
import torch

from Training.BasicDepthModelTrainer import BasicDepthModelTrainer

if __name__ == "__main__":
    print("training model")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    argManager = ArgumentManager(sys.argv)

    validationSetRatio = 0.1
    if validationSetRatio in argManager:
        validationSetRatio = argManager.validationSetRatio

    trainer = BasicDepthModelTrainer(argManager.dataset, device, "NewDepthPredictionModel-SkipConnections", "modelFiles", validationSetRatio, batchSize=1)
    trainer.TrainingLoop()