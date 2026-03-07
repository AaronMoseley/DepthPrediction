import torch

def ScaleInvariantLoss(rgbTensor:torch.Tensor, outputTensor:torch.Tensor, groundTruthTensor:torch.Tensor, validMask:torch.Tensor) -> torch.Tensor:
    validMaskBool = validMask.bool()
    
    groundTruthTensor = groundTruthTensor.clone()
    outputTensor = outputTensor.clone()
    
    elementCount = torch.numel(validMask)

    groundTruthTensor[~validMaskBool] = 0
    outputTensor[~validMaskBool] = 0
    
    logGroundTruth = torch.zeros_like(groundTruthTensor, device=outputTensor.device)
    logGroundTruth[validMaskBool] = torch.log(groundTruthTensor[validMaskBool])

    alpha = torch.zeros_like(logGroundTruth, device=outputTensor.device)
    alpha[validMaskBool] = (logGroundTruth - outputTensor)[validMaskBool]
    alphaSum = torch.sum(alpha, dim=(1, 2, 3))
    alpha = alphaSum / elementCount
    alpha = alpha.view(groundTruthTensor.shape[0], 1, 1, 1)

    difference = torch.zeros_like(outputTensor, device=outputTensor.device)
    difference[validMaskBool] = (outputTensor - logGroundTruth)[validMaskBool]
    difference = difference + alpha
    difference = torch.pow(difference, 2.0)

    differenceSum = torch.sum(difference, dim=(1, 2, 3))
    result = differenceSum / elementCount

    result = torch.mean(result)
    return result

def GradientX(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def GradientY(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def EdgeAwareSmoothnessLoss(rgbTensor:torch.Tensor, outputTensor:torch.Tensor, groundTruthTensor:torch.Tensor, validMask:torch.Tensor):
    """
    depth: (B, 1, H, W)
    image: (B, 3, H, W)
    """
    rgbTensor = rgbTensor.clone()
    outputTensor = outputTensor.clone()

    depth_dx = GradientX(outputTensor)
    depth_dy = GradientY(outputTensor)

    image_dx = GradientX(rgbTensor)
    image_dy = GradientY(rgbTensor)

    # Average color gradient across channels
    image_dx = torch.mean(torch.abs(image_dx), dim=1, keepdim=True)
    image_dy = torch.mean(torch.abs(image_dy), dim=1, keepdim=True)

    # Edge-aware weighting
    weight_x = torch.exp(-image_dx)
    weight_y = torch.exp(-image_dy)

    smoothness_x = depth_dx.abs() * weight_x
    smoothness_y = depth_dy.abs() * weight_y

    return smoothness_x.mean() + smoothness_y.mean()

def EdgeFocusedScaleInvariantLoss(rgbTensor:torch.Tensor, outputTensor:torch.Tensor, groundTruthTensor:torch.Tensor, validMask:torch.Tensor):
    threshold = 0.5
    
    # Sobel filters for gradient computation
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]],
                           dtype=rgbTensor.dtype,
                           device=rgbTensor.device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]],
                           dtype=rgbTensor.dtype,
                           device=rgbTensor.device).view(1, 1, 3, 3)

    # Apply per channel using groups
    channels = rgbTensor.shape[1]
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)

    grad_x = torch.nn.functional.conv2d(rgbTensor, sobel_x, padding=1, groups=channels)
    grad_y = torch.nn.functional.conv2d(rgbTensor, sobel_y, padding=1, groups=channels)

    # Gradient magnitude
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag = torch.mean(grad_mag, dim=1, keepdim=True)

    # Create mask
    mask = grad_mag >= threshold

    edgeFocusedValidMask = validMask.clone()

    edgeFocusedValidMask[~mask] = 0

    resultLoss = ScaleInvariantLoss(rgbTensor, outputTensor, groundTruthTensor, edgeFocusedValidMask)

    return resultLoss