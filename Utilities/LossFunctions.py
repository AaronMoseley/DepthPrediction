import torch

def ScaleInvariantLoss(outputTensor:torch.Tensor, groundTruthTensor:torch.Tensor) -> torch.Tensor:
    logGroundTruth = torch.log(groundTruthTensor)

    alpha = logGroundTruth - outputTensor
    alpha = torch.mean(alpha, (1, 2, 3))
    alpha = alpha.view(1, 3, 1, 1)

    difference = outputTensor - logGroundTruth
    difference = difference + alpha
    difference = torch.pow(difference, 2.0)

    result = torch.mean(difference)
    return result