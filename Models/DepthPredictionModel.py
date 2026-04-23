import torch
import math

class DepthPredictionModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.coarseBranch = CoarseBranch(3, torch.nn.GELU())
		self.fineBranch = FineBranch(3, torch.nn.GELU())
		self.decoder = Decoder(512, torch.nn.ReLU())

	def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
		fineOutput, fineSkip1, fineSkip2, fineSkip3, fineSkip4 = self.fineBranch(inputTensor)
		coarseOutput, coarseSkip1, coarseSkip2, coarseSkip3, coarseSkip4 = self.coarseBranch(inputTensor, fineSkip1, fineSkip2, fineSkip3, fineSkip4)

		upsampledOutput = self.decoder(fineOutput, coarseOutput, coarseSkip1, coarseSkip2, coarseSkip3, coarseSkip4, fineSkip1, fineSkip2, fineSkip3, fineSkip4)

		return upsampledOutput

class CoarseBranch(torch.nn.Module):
	def __init__(self, inputChannels:int=3, finalActivation:torch.nn.Module=None) -> None:
		super().__init__()

		self.finalActivation = finalActivation
		self.inputChannels = inputChannels

		self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

		self.combine1 = torch.nn.Conv2d(64 + 128, 128, kernel_size=1)
		self.combine2 = torch.nn.Conv2d(64 + 128 + 256, 256, kernel_size=1)
		self.combine3 = torch.nn.Conv2d(64 + 128 + 256 + 512, 512, kernel_size=1)

		self.block1 = CoarseBlockWide(inputChannels, 64, False)
		self.block2 = CoarseBlock(64, 128, False)
		self.block3 = CoarseBlock(128, 256, False)
		self.block4 = CoarseBlock(256, 512, False)

	def forward(self, inputTensor:torch.Tensor, fineSkip1:torch.Tensor, fineSkip2:torch.Tensor, fineSkip3:torch.Tensor, fineSkip4:torch.Tensor,) -> tuple[torch.Tensor]:
		skip1 = self.block1(inputTensor)
		encode1 = skip1 + fineSkip1
		encode1 = self.pooling(skip1)

		#resize rgb to encode 1 size
		resize1Shape = (encode1.shape[2], encode1.shape[3])
		resizedInputLayer1 = torch.nn.functional.interpolate(inputTensor, size=resize1Shape, mode="bilinear")

		skip2 = self.block2(resizedInputLayer1, encode1)
		encode2 = skip2 + fineSkip2
		encode2 = self.pooling(skip2)

		resize2Shape = (encode2.shape[2], encode2.shape[3])
		resizedInputLayer2 = torch.nn.functional.interpolate(resizedInputLayer1, size=resize2Shape, mode="bilinear")
		encode1Resized1 = self.pooling(encode1)

		encode3Input = torch.cat([encode1Resized1, encode2], dim=1)
		encode3Input = self.combine1(encode3Input)

		skip3 = self.block3(resizedInputLayer2, encode3Input)
		encode3 = skip3 + fineSkip3
		encode3 = self.pooling(skip3)

		resize3Shape = (encode3.shape[2], encode3.shape[3])
		resizedInputLayer3 = torch.nn.functional.interpolate(resizedInputLayer2, size=resize3Shape, mode="bilinear")
		encode1Resized2 = self.pooling(encode1Resized1)
		encode2Resized1 = self.pooling(encode2)

		encode4Input = torch.cat([encode1Resized2, encode2Resized1, encode3], dim=1)
		encode4Input = self.combine2(encode4Input)

		skip4 = self.block4(resizedInputLayer3, encode4Input)
		encode4 = skip4 + fineSkip4
		encode4 = self.pooling(skip4)

		resize4Shape = (encode4.shape[2], encode4.shape[3])
		encode1Resized3 = self.pooling(encode1Resized2)
		encode2Resized2 = self.pooling(encode2Resized1)
		encode3Resized1 = self.pooling(encode3)

		result = torch.cat([encode1Resized3, encode2Resized2, encode3Resized1, encode4], dim=1)
		result = self.combine3(result)

		if self.finalActivation is not None:
			result = self.finalActivation(result)

		return result, skip1, skip2, skip3, skip4

class FineBranch(torch.nn.Module):
	def __init__(self, inputChannels:int=3, finalActivation:torch.nn.Module=None) -> None:
		super().__init__()

		self.finalActivation = finalActivation
		self.inputChannels = inputChannels

		self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

		self.combine1 = torch.nn.Conv2d(64 + 128, 128, kernel_size=1)
		self.combine2 = torch.nn.Conv2d(64 + 128 + 256, 256, kernel_size=1)
		self.combine3 = torch.nn.Conv2d(64 + 128 + 256 + 512, 512, kernel_size=1)

		self.block1 = FineBlockWide(inputChannels, 64, False)
		self.block2 = FineBlock(64, 128, False)
		self.block3 = FineBlock(128, 256, False)
		self.block4 = FineBlock(256, 512, False)

	def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
		skip1 = self.block1(inputTensor)
		encode1 = self.pooling(skip1)

		#resize rgb to encode 1 size
		resize1Shape = (encode1.shape[2], encode1.shape[3])
		resizedInputLayer1 = torch.nn.functional.interpolate(inputTensor, size=resize1Shape, mode="bilinear")

		skip2 = self.block2(resizedInputLayer1, encode1)
		encode2 = self.pooling(skip2)

		resize2Shape = (encode2.shape[2], encode2.shape[3])
		resizedInputLayer2 = torch.nn.functional.interpolate(resizedInputLayer1, size=resize2Shape, mode="bilinear")
		encode1Resized1 = self.pooling(encode1)

		encode3Input = torch.cat([encode1Resized1, encode2], dim=1)
		encode3Input = self.combine1(encode3Input)

		skip3 = self.block3(resizedInputLayer2, encode3Input)
		encode3 = self.pooling(skip3)

		resize3Shape = (encode3.shape[2], encode3.shape[3])
		resizedInputLayer3 = torch.nn.functional.interpolate(resizedInputLayer2, size=resize3Shape, mode="bilinear")
		encode1Resized2 = self.pooling(encode1Resized1)
		encode2Resized1 = self.pooling(encode2)

		encode4Input = torch.cat([encode1Resized2, encode2Resized1, encode3], dim=1)
		encode4Input = self.combine2(encode4Input)

		skip4 = self.block4(resizedInputLayer3, encode4Input)
		encode4 = self.pooling(skip4)

		resize4Shape = (encode4.shape[2], encode4.shape[3])
		encode1Resized3 = self.pooling(encode1Resized2)
		encode2Resized2 = self.pooling(encode2Resized1)
		encode3Resized1 = self.pooling(encode3)

		result = torch.cat([encode1Resized3, encode2Resized2, encode3Resized1, encode4], dim=1)
		result = self.combine3(result)

		if self.finalActivation is not None:
			result = self.finalActivation(result)

		return result, skip1, skip2, skip3, skip4

class CoarseBlock(torch.nn.Module):
	def __init__(self, inputChannels:int, outputChannels:int, poolResults:bool=True) -> None:
		super().__init__()

		self.inputChannels = inputChannels
		self.outputChannels = outputChannels
		self.useMaxPool = poolResults

		self.kernelSize = 3

		self.activation = torch.nn.GELU()

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
		self.maxPool = torch.nn.AvgPool2d((2, 2))

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

		self.activation = torch.nn.GELU()

		self.norm = torch.nn.GroupNorm(1, outputChannels)

		#convolution
		self.wideKernelH = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.wideKernelSize, 1), groups=inputChannels)
		self.wideKernelV = torch.nn.Conv2d(inputChannels, intermediateChannels, (1, self.wideKernelSize), groups=inputChannels)
		self.centeredConv = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=inputChannels)
		self.mixer = torch.nn.Conv2d(intermediateChannels * 3, outputChannels, kernel_size=1)

		self.conv1 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=outputChannels)
		self.spade1 = SPADEBlock(outputChannels)

		#convolution
		self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=outputChannels)
		self.spade2 = SPADEBlock(outputChannels)

		self.combine1 = torch.nn.Conv2d(outputChannels * 2, outputChannels, kernel_size=1)

		self.conv3 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=outputChannels)
		self.spade3 = SPADEBlock(outputChannels)

		self.combine2 = torch.nn.Conv2d(outputChannels * 3, outputChannels, kernel_size=1)

		#max pool?
		self.maxPool = torch.nn.AvgPool2d((2, 2))

	def forward(self, rgbTensor:torch.Tensor) -> torch.Tensor:
		padV = torch.nn.functional.pad(rgbTensor, (5, 5, 0, 0), mode="reflect")
		padH = torch.nn.functional.pad(rgbTensor, (0, 0, 5, 5), mode="reflect")

		wide1 = self.wideKernelH(padH)
		wide2 = self.wideKernelV(padV)
		center = self.centeredConv(rgbTensor)

		concated = torch.concat([wide1, wide2, center], dim=1)
		mixed = self.mixer(concated)
		normed = self.norm(mixed)
		activated = self.activation(normed)

		convolved1 = self.conv1(activated)
		spade1 = self.spade1(rgbTensor, convolved1)
		activated1 = self.activation(spade1)

		convolved2 = self.conv2(activated1)
		
		combined1 = self.combine1(torch.cat((convolved1, convolved2), dim=1))

		spade2 = self.spade2(rgbTensor, combined1)
		activated2 = self.activation(spade2)

		convolved3 = self.conv3(activated2)

		combined2 = self.combine2(torch.cat((convolved1, convolved2, convolved3), dim=1))

		spade3 = self.spade3(rgbTensor, combined2)
		spade3 = self.norm(spade3)
		result = self.activation(spade3)

		if self.useMaxPool:
			result = self.maxPool(result)

		return result

class FineBlockWide(torch.nn.Module):
	def __init__(self, inputChannels:int, outputChannels:int, useMaxPool:bool) -> None:
		super().__init__()

		self.inputChannels = inputChannels
		self.outputChannels = outputChannels
		self.useMaxPool = useMaxPool

		self.kernelSize = 5
		self.wideKernelSize = 11

		intermediateChannels = self.inputChannels * 16

		self.activation = torch.nn.GELU()

		self.norm = torch.nn.GroupNorm(1, outputChannels)

		#convolution
		self.wideKernelH = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.wideKernelSize, 1), groups=inputChannels)
		self.wideKernelV = torch.nn.Conv2d(inputChannels, intermediateChannels, (1, self.wideKernelSize), groups=inputChannels)
		self.centeredConv = torch.nn.Conv2d(inputChannels, intermediateChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=inputChannels)
		self.mixer = torch.nn.Conv2d(intermediateChannels * 3, outputChannels, kernel_size=1)

		self.conv1 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=outputChannels)
		self.spade1 = SPADEBlock(outputChannels)

		#convolution
		self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=outputChannels)
		self.spade2 = SPADEBlock(outputChannels)

		self.combine1 = torch.nn.Conv2d(outputChannels * 2, outputChannels, kernel_size=1)

		self.conv3 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=outputChannels)
		self.spade3 = SPADEBlock(outputChannels)

		self.combine2 = torch.nn.Conv2d(outputChannels * 3, outputChannels, kernel_size=1)

		#max pool?
		self.maxPool = torch.nn.AvgPool2d((2, 2))

	def forward(self, rgbTensor:torch.Tensor) -> torch.Tensor:
		padV = torch.nn.functional.pad(rgbTensor, (5, 5, 0, 0), mode="reflect")
		padH = torch.nn.functional.pad(rgbTensor, (0, 0, 5, 5), mode="reflect")

		wide1 = self.wideKernelH(padH)
		wide2 = self.wideKernelV(padV)
		center = self.centeredConv(rgbTensor)

		concated = torch.concat([wide1, wide2, center], dim=1)
		mixed = self.mixer(concated)
		normed = self.norm(mixed)
		activated = self.activation(normed)

		convolved1 = self.conv1(activated)
		spade1 = self.spade1(rgbTensor, convolved1)
		activated1 = self.activation(spade1)

		convolved2 = self.conv2(activated1)
		
		combined1 = self.combine1(torch.cat((convolved1, convolved2), dim=1))

		spade2 = self.spade2(rgbTensor, combined1)
		activated2 = self.activation(spade2)

		convolved3 = self.conv3(activated2)

		combined2 = self.combine2(torch.cat((convolved1, convolved2, convolved3), dim=1))

		spade3 = self.spade3(rgbTensor, combined2)
		spade3 = self.norm(spade3)
		result = self.activation(spade3)

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

		self.activation = torch.nn.GELU()

		#convolution
		self.conv1 = torch.nn.Conv2d(inputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=min(inputChannels, outputChannels))
		self.spade1 = SPADEBlock(outputChannels)

		#convolution
		self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=outputChannels)
		self.spade2 = SPADEBlock(outputChannels)

		self.combine1 = torch.nn.Conv2d(outputChannels * 2, outputChannels, kernel_size=1)

		self.conv3 = torch.nn.Conv2d(outputChannels, outputChannels, (self.kernelSize, self.kernelSize), padding=(2, 2), padding_mode="replicate", groups=outputChannels)
		self.spade3 = SPADEBlock(outputChannels)

		self.combine2 = torch.nn.Conv2d(outputChannels * 3, outputChannels, kernel_size=1)

		#max pool?
		self.maxPool = torch.nn.AvgPool2d((2, 2))

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
	
class DecoderBlock(torch.nn.Module):
	def __init__(self, inputChannels:int, outputChannels:int) -> None:
		super().__init__()
		
		self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

		self.kernelSize = 3

		self.norm = torch.nn.GroupNorm(1, outputChannels)

		self.conv1 = torch.nn.Conv2d(inputChannels, outputChannels, kernel_size=(self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=outputChannels)
		self.conv2 = torch.nn.Conv2d(outputChannels, outputChannels, kernel_size=(self.kernelSize, self.kernelSize), padding=(1, 1), padding_mode="replicate", groups=outputChannels)

		self.activation = torch.nn.GELU()

	def forward(self, inputTensor:torch.Tensor, skipConnection:torch.Tensor=None) -> torch.Tensor:
		upsampled = self.upsample(inputTensor)
		
		if skipConnection is not None:
			upsampled = upsampled + skipConnection
		
		convolved1 = self.conv1(upsampled)

		convolved2 = self.conv2(convolved1)

		normed = self.norm(convolved2)
		result = self.activation(normed)

		return result

class Decoder(torch.nn.Module):
	def __init__(self, inputChannels:int, activation:torch.nn.Module=None) -> None:
		super().__init__()

		self.inputChannels = inputChannels
		self.finalActivation = activation

		self.finalConv = torch.nn.Conv2d(32, 1, kernel_size=1)

		self.mixer1 = DecoderMixerBlock(512, kernelSize=5)
		self.mixer2 = DecoderMixerBlock(512, kernelSize=5)

		self.decode1 = DecoderBlock(512, 256)
		self.decode2 = DecoderBlock(256, 128)
		self.decode3 = DecoderBlock(128, 64)
		self.decode4 = DecoderBlock(64, 32)

	def forward(self, coarseOutput:torch.Tensor, fineOutput:torch.Tensor, 
	coarseSkip1:torch.Tensor, coarseSkip2:torch.Tensor, coarseSkip3:torch.Tensor, coarseSkip4:torch.Tensor,
	fineSkip1:torch.Tensor, fineSkip2:torch.Tensor, fineSkip3:torch.Tensor, fineSkip4:torch.Tensor) -> torch.Tensor:
		
		mixed1 = self.mixer1(coarseOutput)
		mixed2 = self.mixer2(mixed1)

		decoded1 = self.decode1(mixed2, coarseSkip4)
		decoded2 = self.decode2(decoded1, coarseSkip3)
		decoded3 = self.decode3(decoded2, coarseSkip2)
		decoded4 = self.decode4(decoded3, coarseSkip1)

		result = self.finalConv(decoded4)

		if self.finalActivation is not None:
			result = self.finalActivation(result)

		return result

class DecoderMixerBlock(torch.nn.Module):
	def __init__(self, inputChannels:int, kernelSize:int) -> None:
		super().__init__()

		self.inputChannels = inputChannels
		self.kernelSize = kernelSize

		self.wideKernel = torch.nn.Conv2d(inputChannels, inputChannels, kernel_size=self.kernelSize, groups=inputChannels, padding=self.kernelSize // 2, padding_mode="reflect")
		self.mixer = torch.nn.Conv2d(inputChannels, inputChannels, kernel_size=1)
		self.norm = torch.nn.GroupNorm(1, inputChannels)
		self.activation = torch.nn.GELU()

	def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
		storedInput = inputTensor

		conv1 = self.wideKernel(inputTensor)
		activated = self.activation(conv1)
		normed = self.norm(activated)

		normed = normed + storedInput

		mixed = self.mixer(normed)
		activated2 = self.activation(mixed)
		normed2 = self.norm(activated2)

		return normed2
	
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