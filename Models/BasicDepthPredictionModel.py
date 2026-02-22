import torch
import math

class DepthPredictionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.coarseBranch = CoarseBranch(3, torch.nn.Sigmoid())
        self.fineBranch = FineBranch(3, torch.nn.Sigmoid())
        self.decoder = Decoder(64, torch.nn.ReLU())

    def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
        coarseOutput = self.coarseBranch(inputTensor)
        fineOutput = self.fineBranch(inputTensor, coarseOutput)

        upsampledOutput = self.decoder(fineOutput)

        return upsampledOutput

class CoarseBranch(torch.nn.Module):
    def __init__(self, inputChannels:int=3, finalActivation:torch.nn.Module=None) -> None:
        super().__init__()

        self.finalActivation = finalActivation
        self.inputChannels = inputChannels

        self.block1 = CoarseBlockWide(inputChannels, 64, True)
        self.block2 = CoarseBlock(64, 128, False)
        self.block3 = CoarseBlock(128, 256, False)
        self.block4 = CoarseBlock(256, 256, True)

        self.finalConv = torch.nn.Conv2d(256, 1, (1, 1))

    def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
        out1 = self.block1(inputTensor)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        result = self.finalConv(out4)

        if self.finalActivation is not None:
            result = self.finalActivation(result)

        return result

class FineBranch(torch.nn.Module):
    def __init__(self, inputChannels:int=3, finalActivation:torch.nn.Module=None) -> None:
        super().__init__()

        self.finalActivation = finalActivation
        self.inputChannels = inputChannels

        self.block1 = FineBlockWide(inputChannels, 63, True)
        self.block2 = FineBlock(64, 64, False)
        self.block3 = FineBlock(64, 64, False)

    def forward(self, inputTensor:torch.Tensor, coarseOutput:torch.Tensor) -> torch.Tensor:
        out1 = self.block1(inputTensor)

        concatInput = torch.concat((out1, coarseOutput), dim=1)

        out2 = self.block2(concatInput)
        result = self.block3(out2)

        if self.finalActivation is not None:
            result = self.finalActivation(result)

        return result

class CoarseBlock(torch.nn.Module):
    def __init__(self, inputChannels:int, outputChannels:int, poolResults:bool=True) -> None:
        super().__init__()

        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.useMaxPool = poolResults

        self.kernelSize = 3

        #convolution
        self.conv1 = torch.nn.Conv2d(inputChannels, outputChannels, (self.kernelSize, self.kernelSize), groups=self.inputChannels, padding=(1, 1))

        #layer norm
        self.norm1 = torch.nn.GroupNorm(1, outputChannels)

        #gelu
        self.activation1 = torch.nn.GELU()

        #convolution
        self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), groups=self.outputChannels, padding=(1, 1))

        #layer norm
        self.norm2 = torch.nn.GroupNorm(1, outputChannels)

        #max pool?
        self.maxPool = torch.nn.MaxPool2d((2, 2))

    def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
        convolved1 = self.conv1(inputTensor)
        normalized1 = self.norm1(convolved1)
        activated1 = self.activation1(normalized1)

        convolved2 = self.conv2(activated1)
        result = self.norm2(convolved2)

        if self.useMaxPool:
            result = self.maxPool(result)

        return result

class CoarseBlockWide(torch.nn.Module):
    def __init__(self, inputChannels:int, outputChannels:int, poolResults:bool=True) -> None:
        super().__init__()

        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.useMaxPool = poolResults

        self.kernelSize = 3
        self.wideKernelSize = 11

        intermediateChannels = inputChannels * 16

        #convolution
        self.wideKernelH = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.wideKernelSize, 1), groups=self.inputChannels)
        self.wideKernelV = torch.nn.Conv2d(inputChannels, intermediateChannels, (1, self.wideKernelSize), groups=self.inputChannels)
        self.centeredConv = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.kernelSize, self.kernelSize), groups=self.inputChannels, padding=(1, 1))
        self.mixer = torch.nn.Conv2d(intermediateChannels * 3, outputChannels, (1, 1))

        #layer norm
        self.norm1 = torch.nn.GroupNorm(1, outputChannels)

        #gelu
        self.activation1 = torch.nn.GELU()

        #convolution
        self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), groups=self.outputChannels, padding=(1, 1))

        #layer norm
        self.norm2 = torch.nn.GroupNorm(1, outputChannels)

        #max pool?
        self.maxPool = torch.nn.MaxPool2d((2, 2))

    def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
        padV = torch.nn.functional.pad(inputTensor, (5, 5, 0, 0), mode="reflect")
        padH = torch.nn.functional.pad(inputTensor, (0, 0, 5, 5), mode="reflect")

        wide1 = self.wideKernelH(padH)
        wide2 = self.wideKernelV(padV)
        center = self.centeredConv(inputTensor)

        concated = torch.concat([wide1, wide2, center], dim=1)
        convolved1 = self.mixer(concated)

        normalized1 = self.norm1(convolved1)
        activated1 = self.activation1(normalized1)

        convolved2 = self.conv2(activated1)
        result = self.norm2(convolved2)

        if self.useMaxPool:
            result = self.maxPool(result)

        return result

class FineBlockWide(torch.nn.Module):
    def __init__(self, inputChannels:int, outputChannels:int, poolResults:bool=True) -> None:
        super().__init__()

        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.useMaxPool = poolResults

        self.kernelSize = 5
        self.wideKernelSize = 9

        intermediateChannels = self.inputChannels * 16

        #convolution
        self.wideKernelH = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.wideKernelSize, 1), groups=self.inputChannels)
        self.wideKernelV = torch.nn.Conv2d(inputChannels, intermediateChannels, (1, self.wideKernelSize), groups=self.inputChannels)
        self.centeredConv = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.kernelSize, self.kernelSize), groups=self.inputChannels, padding=(2, 2))
        self.mixer = torch.nn.Conv2d(intermediateChannels * 3, outputChannels, (1, 1))

        #layer norm
        self.norm1 = torch.nn.GroupNorm(1, outputChannels)

        #gelu
        self.activation1 = torch.nn.GELU()

        #convolution
        self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), groups=self.outputChannels, stride=2, padding=(2, 2))

        #layer norm
        self.norm2 = torch.nn.GroupNorm(1, outputChannels)

        #max pool?
        self.maxPool = torch.nn.MaxPool2d((2, 2))

    def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
        padV = torch.nn.functional.pad(inputTensor, (4, 4, 0, 0), mode="reflect")
        padH = torch.nn.functional.pad(inputTensor, (0, 0, 4, 4), mode="reflect")

        wide1 = self.wideKernelH(padH)
        wide2 = self.wideKernelV(padV)
        center = self.centeredConv(inputTensor)

        concated = torch.cat([wide1, wide2, center], dim=1)
        convolved1 = self.mixer(concated)

        normalized1 = self.norm1(convolved1)
        activated1 = self.activation1(normalized1)

        convolved2 = self.conv2(activated1)
        result = self.norm2(convolved2)

        if self.useMaxPool:
            result = self.maxPool(result)

        return result

class FineBlock(torch.nn.Module):
    def __init__(self, inputChannels:int, outputChannels:int, poolResults:bool=True) -> None:
        super().__init__()

        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.useMaxPool = poolResults

        self.kernelSize = 5

        #convolution
        self.conv1 = torch.nn.Conv2d(inputChannels, outputChannels, (self.kernelSize, self.kernelSize), groups=self.inputChannels, padding=(2, 2))

        #layer norm
        self.norm1 = torch.nn.GroupNorm(1, outputChannels)

        #gelu
        self.activation1 = torch.nn.GELU()

        #convolution
        self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), groups=self.outputChannels, padding=(2, 2))

        #layer norm
        self.norm2 = torch.nn.GroupNorm(1, outputChannels)

        #max pool?
        self.maxPool = torch.nn.MaxPool2d((2, 2))

    def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
        convolved1 = self.conv1(inputTensor)
        normalized1 = self.norm1(convolved1)
        activated1 = self.activation1(normalized1)

        convolved2 = self.conv2(activated1)
        result = self.norm2(convolved2)

        if self.useMaxPool:
            result = self.maxPool(result)

        return result
    
class Decoder(torch.nn.Module):
    def __init__(self, inputChannels:int, activation:torch.nn.Module=None) -> None:
        super().__init__()

        self.inputChannels = inputChannels
        self.activation = activation

        self.kernelSize = 3

        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv1 = torch.nn.Conv2d(inputChannels, inputChannels // 2, kernel_size=(self.kernelSize, self.kernelSize), padding=(1, 1), groups=inputChannels // 2)
        self.conv2 = torch.nn.Conv2d(inputChannels // 2, inputChannels // 4, kernel_size=(self.kernelSize, self.kernelSize), padding=(1, 1), groups=inputChannels // 4)

        self.finalConv = torch.nn.Conv2d(inputChannels // 4, 1, kernel_size=(1, 1))

    def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
        convolved1 = self.conv1(inputTensor)
        upsampled1 = self.upsample(convolved1)

        convolved2 = self.conv2(upsampled1)
        upsampled2 = self.upsample(convolved2)

        result = self.finalConv(upsampled2)

        if self.activation is not None:
            result = self.activation(result)

        return result