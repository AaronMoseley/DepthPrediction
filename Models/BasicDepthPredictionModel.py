import torch
import math

class DepthPredictionModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.coarseBranch = CoarseBranch(3, torch.nn.LeakyReLU())
		self.fineBranch = FineBranch(3, torch.nn.Sigmoid())
		self.decoder = Decoder(128, torch.nn.ReLU())

	def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
		coarseOutput, skip1, skip2, skip3 = self.coarseBranch(inputTensor)
		fineOutput, fineSkipConnection = self.fineBranch(inputTensor, coarseOutput, skip2, skip3)

		upsampledOutput = self.decoder(fineOutput, coarseOutput, fineSkipConnection, skip1)

		return upsampledOutput

class CoarseBranch(torch.nn.Module):
	def __init__(self, inputChannels:int=3, finalActivation:torch.nn.Module=None) -> None:
		super().__init__()

		self.finalActivation = finalActivation
		self.inputChannels = inputChannels

		self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

		self.combine1 = torch.nn.Conv2d(64 + 128, 128, kernel_size=1)
		self.combine2 = torch.nn.Conv2d(64 + 128 + 256, 256, kernel_size=1)
		self.combine3 = torch.nn.Conv2d(64 + 128 + 256 + 128, 128, kernel_size=1)

		self.block1 = CoarseBlockWide(inputChannels, 64, True)
		self.block2 = CoarseBlock(64, 128, False)
		self.block3 = CoarseBlock(128, 256, False)
		self.block4 = CoarseBlock(256, 128, True)

	def forward(self, inputTensor:torch.Tensor) -> tuple[torch.Tensor]:
		encode1 = self.block1(inputTensor)

		#resize rgb to encode 1 size
		resize1Shape = (encode1.shape[2], encode1.shape[3])
		resizedInputLayer1 = torch.nn.functional.interpolate(inputTensor, size=resize1Shape, mode="bilinear")

		encode2 = self.block2(resizedInputLayer1, encode1)

		encode3Input = torch.cat([encode1, encode2], dim=1)
		encode3Input = self.combine1(encode3Input)

		encode3 = self.block3(resizedInputLayer1, encode3Input)

		encode4Input = torch.cat([encode1, encode2, encode3], dim=1)
		encode4Input = self.combine2(encode4Input)

		encode4 = self.block4(resizedInputLayer1, encode4Input)

		encode1DownSampled1 = self.pooling(encode1)
		encode2DownSampled1 = self.pooling(encode2)
		encode3DownSampled1 = self.pooling(encode3)
		result = torch.cat([encode1DownSampled1, encode2DownSampled1, encode3DownSampled1, encode4], dim=1)
		result = self.combine3(result)

		if self.finalActivation is not None:
			result = self.finalActivation(result)

		return result, encode1, encode3, encode4

class FineBranch(torch.nn.Module):
	def __init__(self, inputChannels:int=3, finalActivation:torch.nn.Module=None) -> None:
		super().__init__()

		self.finalActivation = finalActivation
		self.inputChannels = inputChannels

		self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

		self.combine1 = torch.nn.Conv2d(128 + 128, 128, kernel_size=1)
		self.combine2 = torch.nn.Conv2d(128 + 128 + 256, 256, kernel_size=1)
		self.combine3 = torch.nn.Conv2d(128 + 128 + 256 + 128, 128, kernel_size=1)

		self.block1 = FineBlockWide(inputChannels, 128, True)
		self.block2 = FineBlock(256, 128, False)
		self.block3 = FineBlock(128, 256, False)
		self.block4 = FineBlock(256, 128, False)

	def forward(self, inputTensor:torch.Tensor, coarseOutput:torch.Tensor, skip1:torch.Tensor, skip2:torch.Tensor) -> torch.Tensor:
		encode1 = self.block1(inputTensor)

		concatInput = torch.concat((encode1, coarseOutput), dim=1)

		encode2 = self.block2(concatInput)

		combined1 = self.combine1(torch.cat((encode1, encode2), dim=1))

		encode3 = self.block3(combined1)
		resizedSkip1 = torch.nn.functional.interpolate(skip1, size=(encode3.shape[2], encode3.shape[3]), mode="bilinear")
		encode3 = encode3 + resizedSkip1

		combined2 = self.combine2(torch.cat((encode1, encode2, encode3), dim=1))

		encode4 = self.block4(combined2)
		encode4 = encode4 + skip2

		result = self.combine3(torch.cat((encode1, encode2, encode3, encode4), dim=1))

		if self.finalActivation is not None:
			result = self.finalActivation(result)

		return result, concatInput

class CoarseBlock(torch.nn.Module):
	def __init__(self, inputChannels:int, outputChannels:int, poolResults:bool=True) -> None:
		super().__init__()

		self.inputChannels = inputChannels
		self.outputChannels = outputChannels
		self.useMaxPool = poolResults

		self.kernelSize = 3

		self.activation = torch.nn.LeakyReLU()

		#convolution
		self.conv1 = torch.nn.Conv2d(inputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=min(inputChannels, outputChannels))
		self.spade1 = SPADEBlock(outputChannels)

		#convolution
		self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=outputChannels)
		self.spade2 = SPADEBlock(outputChannels)

		self.combine1 = torch.nn.Conv2d(outputChannels * 2, outputChannels, kernel_size=1)

		self.conv3 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=outputChannels)
		self.spade3 = SPADEBlock(outputChannels)

		self.combine2 = torch.nn.Conv2d(outputChannels * 3, outputChannels, kernel_size=1)

		#max pool?
		self.maxPool = torch.nn.MaxPool2d((2, 2))

	def forward(self, rgbTensor:torch.Tensor, inputTensor:torch.Tensor) -> torch.Tensor:
		convolved1 = self.conv1(inputTensor)
		spade1 = self.spade1(rgbTensor, convolved1)
		activated1 = self.activation(spade1)

		convolved2 = self.conv2(activated1)
		
		combined1 = self.combine1(torch.cat((convolved1, convolved2), dim=1))

		spade2 = self.spade2(rgbTensor, combined1)
		activated2 = self.activation(spade2)

		convolved3 = self.conv3(activated2)

		combined2 = self.combine2(torch.cat((convolved1, convolved2, convolved3), dim=1))

		spade3 = self.spade3(rgbTensor, combined2)
		result = self.activation(spade3)

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

		self.activation = torch.nn.LeakyReLU()

		#convolution
		self.wideKernelH = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.wideKernelSize, 1), groups=inputChannels)
		self.wideKernelV = torch.nn.Conv2d(inputChannels, intermediateChannels, (1, self.wideKernelSize), groups=inputChannels)
		self.centeredConv = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=inputChannels)
		self.mixer = torch.nn.Conv2d(intermediateChannels * 3, outputChannels, kernel_size=1)

		self.spade1 = SPADEBlock(outputChannels)

		#convolution
		self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=outputChannels)
		self.spade2 = SPADEBlock(outputChannels)

		self.combine1 = torch.nn.Conv2d(outputChannels * 2, outputChannels, kernel_size=1)

		self.conv3 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=outputChannels)
		self.spade3 = SPADEBlock(outputChannels)

		self.combine2 = torch.nn.Conv2d(outputChannels * 3, outputChannels, kernel_size=1)

		#max pool?
		self.maxPool = torch.nn.MaxPool2d((2, 2))

	def forward(self, rgbTensor:torch.Tensor) -> torch.Tensor:
		padV = torch.nn.functional.pad(rgbTensor, (5, 5, 0, 0), mode="reflect")
		padH = torch.nn.functional.pad(rgbTensor, (0, 0, 5, 5), mode="reflect")

		wide1 = self.wideKernelH(padH)
		wide2 = self.wideKernelV(padV)
		center = self.centeredConv(rgbTensor)

		concated = torch.concat([wide1, wide2, center], dim=1)
		convolved1 = self.mixer(concated)

		spade1 = self.spade1(rgbTensor, convolved1)
		activated1 = self.activation(spade1)

		convolved2 = self.conv2(activated1)
		
		combined1 = self.combine1(torch.cat((convolved1, convolved2), dim=1))

		spade2 = self.spade2(rgbTensor, combined1)
		activated2 = self.activation(spade2)

		convolved3 = self.conv3(activated2)

		combined2 = self.combine2(torch.cat((convolved1, convolved2, convolved3), dim=1))

		spade3 = self.spade3(rgbTensor, combined2)
		result = self.activation(spade3)

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

		self.activation = torch.nn.Sigmoid()

		#convolution
		self.wideKernelH = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.wideKernelSize, 1), groups=inputChannels, stride=2)
		self.wideKernelV = torch.nn.Conv2d(inputChannels, intermediateChannels, (1, self.wideKernelSize), groups=inputChannels, stride=2)
		self.centeredConv = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=inputChannels, stride=2)
		self.mixer = torch.nn.Conv2d(intermediateChannels * 3, outputChannels, kernel_size=1)

		#convolution
		self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=outputChannels)

		self.conv3 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=outputChannels)

		#max pool?
		self.maxPool = torch.nn.AvgPool2d((2, 2))

	def forward(self, rgbTensor:torch.Tensor) -> torch.Tensor:
		padV = torch.nn.functional.pad(rgbTensor, (4, 4, 0, 0), mode="reflect")
		padH = torch.nn.functional.pad(rgbTensor, (0, 0, 4, 4), mode="reflect")

		wide1 = self.wideKernelH(padH)
		wide2 = self.wideKernelV(padV)
		center = self.centeredConv(rgbTensor)

		concated = torch.concat([wide1, wide2, center], dim=1)
		convolved1 = self.mixer(concated)

		activated1 = self.activation(convolved1)

		convolved2 = self.conv2(activated1)
		
		activated2 = self.activation(convolved2)

		convolved3 = self.conv3(activated2)

		result = self.activation(convolved3)

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

		self.activation = torch.nn.Sigmoid()

		#convolution
		self.conv1 = torch.nn.Conv2d(inputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=min(inputChannels, outputChannels))

		#convolution
		self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=outputChannels)

		self.conv3 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=outputChannels)

		#max pool?
		self.maxPool = torch.nn.AvgPool2d((2, 2))

	def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
		convolved1 = self.conv1(inputTensor)
		activated1 = self.activation(convolved1)

		convolved2 = self.conv2(activated1)
		
		activated2 = self.activation(convolved2)

		convolved3 = self.conv3(activated2)

		result = self.activation(convolved3)

		if self.useMaxPool:
			result = self.maxPool(result)

		return result
	
class Decoder(torch.nn.Module):
	def __init__(self, inputChannels:int, activation:torch.nn.Module=None) -> None:
		super().__init__()

		self.inputChannels = inputChannels
		self.finalActivation = activation

		self.kernelSize = 3

		self.downsampleChannel = torch.nn.Conv2d(256, 64, kernel_size=1)

		self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
		self.activation = torch.nn.Sigmoid()

		self.conv1 = torch.nn.Conv2d(inputChannels, inputChannels // 2, kernel_size=(self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=inputChannels // 2)
		self.norm1 = torch.nn.GroupNorm(1, inputChannels // 2)

		self.conv2 = torch.nn.Conv2d(inputChannels // 2, inputChannels // 4, kernel_size=(self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=inputChannels // 4)
		self.norm2 = torch.nn.GroupNorm(1, inputChannels // 4)

		self.finalConv = torch.nn.Conv2d(inputChannels // 4, 1, kernel_size=1)

	def forward(self, inputTensor:torch.Tensor, coarseOutput:torch.Tensor, fineSkipConnection:torch.Tensor, coarseSkipConnection:torch.Tensor) -> torch.Tensor:
		inputTensor = inputTensor + coarseOutput
		
		upsampled1 = self.upsample(inputTensor)
		convolved1 = self.conv1(upsampled1)
		normed1 = self.norm1(convolved1)
		activated1 = self.activation(normed1)

		resizeShape = (activated1.shape[2], activated1.shape[3])
		resizedSkipConnection = torch.nn.functional.interpolate(fineSkipConnection, size=resizeShape, mode="bilinear")
		resizedSkipConnection = self.downsampleChannel(resizedSkipConnection)

		activated1 = activated1 + resizedSkipConnection + coarseSkipConnection

		upsampled2 = self.upsample(activated1)
		convolved2 = self.conv2(upsampled2)
		normed2 = self.norm2(convolved2)
		activated2 = self.activation(normed2)

		result = self.finalConv(activated2)

		if self.finalActivation is not None:
			result = self.finalActivation(result)

		return result
	
class SPADEBlock(torch.nn.Module):
	def __init__(self, inChannels:int):
		super().__init__()
		
		self.inChannels = inChannels

		self.norm = torch.nn.GroupNorm(1, inChannels)

		self.rgbEmbeddingConv = torch.nn.Conv2d(3, 128, kernel_size=3)

		self.attentionMaskConv = torch.nn.Conv2d(128, self.inChannels, kernel_size=3)
		self.additiveConnectionConv = torch.nn.Conv2d(128, self.inChannels, kernel_size=3)

		self.padding = torch.nn.ReflectionPad2d((1, 1, 1, 1))

	def forward(self, rgbTensor:torch.Tensor, inputTensor:torch.Tensor):
		paddedRGB = self.padding(rgbTensor)

		#layer norm on input tensor
		normed = self.norm(inputTensor)

		#128-channel dw conv on resized rgb tensor
		rgbEmbedding = self.rgbEmbeddingConv(paddedRGB)
		rgbEmbedding = self.padding(rgbEmbedding)

		#get attention mask with 3x3 dw conv on embedded rgb
		attentionMask = self.attentionMaskConv(rgbEmbedding)

		#get additive connection with 3x3 conv on embedded rgb
		additive = self.additiveConnectionConv(rgbEmbedding)

		#combine input with mask and additive connection
		output = normed * attentionMask
		output = output + additive

		return output