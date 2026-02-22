import torch

def ScaleInvariantLoss(outputTensor:torch.Tensor, groundTruthTensor:torch.Tensor, validMask:torch.Tensor) -> torch.Tensor:
    groundTruthTensor = groundTruthTensor.clone()
    outputTensor = outputTensor.clone()
    
    groundTruthTensor[validMask < 0.01] = 0
    outputTensor[validMask < 0.01] = 0
    
    logGroundTruth = torch.log(groundTruthTensor)

    alpha = logGroundTruth - outputTensor
    alpha = torch.mean(alpha, (1, 2, 3))
    alpha = alpha.view(1, 3, 1, 1)

    difference = outputTensor - logGroundTruth
    difference = difference + alpha
    difference = torch.pow(difference, 2.0)

    result = torch.mean(difference)
    return result