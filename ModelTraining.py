import sys
from Utilities.ArgumentManager import ArgumentManager
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Datasets.DepthPredictionDataset import DepthPredictionDataset
from Models.BasicDepthPredictionModel import DepthPredictionModel
from Utilities.LossFunctions import ScaleInvariantLoss

def TrainModel(model:torch.nn.Module, lossFunction, trainingDataset:Dataset, validationDataset:Dataset, batchSize:int=3, learnRate:float=0.001, epochs:int=100) -> None:
    #create training dataloader
    trainingLoader = torch.utils.data.DataLoader(trainingDataset, batch_size=batchSize, shuffle=True)

    #create validation dataloader
    validationLoader = torch.utils.data.DataLoader(validationDataset, batch_size=batchSize, shuffle=False)
    
    #create optimizer
    optimizer = torch.optim.RAdam(model.parameters(), learnRate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10)

    trainBatchCount = len(trainingLoader)
    validationBatchCount = len(validationLoader)

    #loop through epochs
    for epochIndex in range(epochs):
        #set model to train mode
        model.train()

        print(f"Starting epoch {epochIndex}")

        #loss = 0 for recording per-epoch loss
        trainingLoss = 0.0
        #loop through training batches
        for batchIndex, (inputTensor, gtTensor) in tqdm(enumerate(trainingLoader)):
            optimizer.zero_grad()
            
            #get output
            outputTensor = model(inputTensor)

            #get loss
            loss:torch.Tensor = lossFunction(outputTensor, gtTensor)

            #apply loss
            loss.backward()
            optimizer.step()

            #step scheduler if needed
            scheduler.step(epochIndex + (batchIndex / trainBatchCount))

            trainingLoss += loss.detach().item()

        #set model to eval mode
        model.eval()

        #validation loss = 0 for recording
        validationLoss = 0.0
        #loop through validation batches
        for batchIndex, (inputTensor, gtTensor) in tqdm(enumerate(validationLoader)):
            #get output
            outputTensor = model(inputTensor)

            #get loss
            loss:torch.Tensor = lossFunction(outputTensor, gtTensor)

            #add to total
            validationLoss += loss.detach().item()

        #report epoch index and train/validation loss
        averageTrainingLoss = trainingLoss / trainBatchCount
        averageValidationLoss = validationLoss / validationBatchCount
        print(f"Epoch {epochIndex}:\n\tTraining Loss: {averageTrainingLoss}\n\tValidation Loss: {averageValidationLoss}\n")

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

    model = DepthPredictionModel().to(device)

    TrainModel(model, ScaleInvariantLoss, trainingDataset, validationDataset)