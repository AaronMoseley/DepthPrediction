import torch
import math

class UNetDepthPredictionModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.encoder = EncoderOnlyModelSPADEDenseWideStartAvgPool()
		self.decoder = BasicDecoderDWGLNBmix(channelNames=["depth"], activationFunction=torch.nn.functional.relu)

		total_params = sum(p.numel() for p in self.parameters())

		pass

	def forward(self, inputTensor:torch.Tensor) -> torch.Tensor:
		bridge, skip1, skip2, skip3, skip4 = self.encoder(inputTensor)
		decoderOut = self.decoder(bridge, skip1, skip2, skip3, skip4)
		return decoderOut
	
class BasicDecoderDWGLNBmix(torch.nn.Module):
	def __init__(self, channelNames = None, activationFunction = None):
		super().__init__()
		self.Name="DecoderDWGeluLayerNormB512"

		if channelNames is not None:
			self.outputChannelCount = len(channelNames)
		else:
			self.outputChannelCount = 3 # assume RGB.

		self.mixBridge512 = MixerBlockDWGLN(512, kernel_size=5, padmode='reflect')
		self.mixBridge512_2 = MixerBlockDWGLN(512, kernel_size=5, padmode='reflect')
		self.decode1 = DecoderBlockResidualDWGLN(512, 256) 
		self.decode2 = DecoderBlockResidualDWGLN(256, 128)
		self.decode3 = DecoderBlockResidualDWGLN(128, 64)
		self.decode4 = DecoderBlockResidualDWGLN(64, 64)
		self.final = DepthWiseConv(64, self.outputChannelCount, kernel_size = 1)
		self.finalActivation = None
		if activationFunction is not None:
			self.finalActivation = activationFunction

	def forward(self, bridge, skip1, skip2, skip3, skip4):
		mixed = self.mixBridge512(bridge)
		mixed2 = self.mixBridge512_2(mixed)
		decode1 = self.decode1(mixed2, skip4)  
		decode2 = self.decode2(decode1, skip3)
		decode3 = self.decode3(decode2, skip2)
		decode4 = self.decode4(decode3, skip1)
		final = self.final(decode4)
		
		if self.finalActivation is not None:
			final = self.finalActivation(final)

		return final	
	
class DecoderBlockResidualDWGLN(torch.nn.Module):
	def __init__(self, inFilterCount, outFilterCount):
		super().__init__()
		self.channelCountOut = outFilterCount
		# using upsample+conv instead of transpose conv, as this reduces checkerboard artifacts according to recent literature.
		# https://distill.pub/2016/deconv-checkerboard/
		self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		#self.projectInToOut = DepthWiseConvWithPadding(channelsIn=inFilterCount, channelsOut=outFilterCount, kernel_size=3)
		self.commonBlock = CommonConvBlockDWGLN(inFilterCount, outFilterCount)

	def forward(self, inputLayer, skipLayer):
		up1 = self.upsample(inputLayer)  # resize, keeping channel count of input.
		# try add instead of concat, similar to resnet.
		combine = up1 + skipLayer
		conv1 = self.commonBlock(combine)  # reproject to smaller channel count.
		return conv1

class CommonConvBlockDWGLN(torch.nn.Module):
	def __init__(self, channelsIn, filterCount, kernel_size=3, padmode='replicate'):
		super().__init__()
		self.channelCountOut = filterCount
		self.kernel_size = kernel_size
		p = self.padding_required = kernel_size // 2
		self.paddingTuple = (p,p,p,p)
		self.pad_mode = padmode
		if padmode == 'replicate':
			padder = torch.nn.ReflectionPad2d
		elif padmode == 'reflect':
			padder = torch.nn.ReflectionPad2d

		self.layers = torch.nn.Sequential(
			padder(self.paddingTuple),
			DepthWiseConv(channelsIn, filterCount, kernel_size=kernel_size, stride=1),
			ChannelFirstLayerNorm(filterCount),
			torch.nn.GELU(),
			padder(self.paddingTuple),
			DepthWiseConv(filterCount, filterCount, kernel_size=kernel_size, stride=1),
			ChannelFirstLayerNorm(filterCount)
		)
		
	def forward(self, inputLayer):
		result = self.layers(inputLayer)
		return result

class MixerBlockDWGLN(torch.nn.Module):
	def __init__(self, channelCount, kernel_size:int=7, padmode='reflect'):
		super().__init__()
		self.kernel_size = kernel_size
		p = self.padding_required = kernel_size // 2
		self.paddingTuple = (p,p,p,p)
		self.pad_mode = padmode
		if padmode == 'replicate':
			padder = torch.nn.ReflectionPad2d
		elif padmode == 'reflect':
			padder = torch.nn.ReflectionPad2d

		self.wideKernel = torch.nn.Conv2d(channelCount, channelCount, kernel_size=kernel_size, groups=channelCount)
		self.mixer = torch.nn.Conv2d(channelCount, channelCount, kernel_size=1)
		self.norm1 = ChannelFirstLayerNorm(channelCount)
		self.norm2 =  ChannelFirstLayerNorm(channelCount)
		self.padder = padder(self.paddingTuple)

	def forward(self, inputLayer):
			
		storedInput = inputLayer

		# grab nearby patches using wide kernel.
		padded = self.padder(inputLayer)
		c1 = self.wideKernel(padded)
		act = torch.nn.functional.gelu(c1)
		norm1 = self.norm1(act)
		residual = storedInput + norm1

		# mix patches together using 1x1 kernel.
		mixed = self.mixer(residual)
		mixedActivated = torch.nn.functional.gelu(mixed)
		mixedNorm = self.norm2(mixedActivated)

		return mixedNorm

class EncoderOnlyModelSPADEDenseWideStartAvgPool(torch.nn.Module):
	def __init__(self, inputChannels:int=3):
		super().__init__()

		self.Name = "EncoderDWGLN"
		
		self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

		self.combine1 = torch.nn.Conv2d(64 + 128, 128, kernel_size=1)
		self.combine2 = torch.nn.Conv2d(64 + 128 + 256, 256, kernel_size=1)
		self.combine3 = torch.nn.Conv2d(64 + 128 + 256 + 512, 512, kernel_size=1)

		self.encode1 = EncoderBlockSPADEWideStart(inputChannels, 64)
		self.encode2 = EncoderBlockSPADE(64, 128)
		self.encode3 = EncoderBlockSPADE(128, 256)
		self.encode4 = EncoderBlockSPADE(256, 512)  # note, results in 16x16
		
	def Freeze(self):
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, inputLayer):
		encode1, skip1 = self.encode1(inputLayer) 

		#resize rgb to encode 1 size
		resize1Shape = (encode1.shape[2], encode1.shape[3])
		resizedInputLayer1 = torch.nn.functional.interpolate(inputLayer, size=resize1Shape, mode="bilinear")

		encode2, skip2 = self.encode2(resizedInputLayer1, encode1)  

		#resize rgb to encode 2 size
		resize2Shape = (encode2.shape[2], encode2.shape[3])
		resizedInputLayer2 = torch.nn.functional.interpolate(resizedInputLayer1, size=resize2Shape, mode="bilinear")

		encode1DownSampled1 = self.pooling(encode1)
		encode3Input = torch.cat([encode1DownSampled1, encode2], dim=1)
		encode3Input = self.combine1(encode3Input)

		encode3, skip3 = self.encode3(resizedInputLayer2, encode3Input)   

		#resize rgb to encode 3 size
		resize3Shape = (encode3.shape[2], encode3.shape[3])
		resizedInputLayer3 = torch.nn.functional.interpolate(resizedInputLayer2, size=resize3Shape, mode="bilinear")

		encode1DownSampled2 = self.pooling(encode1DownSampled1)
		encode2DownSampled1 = self.pooling(encode2)
		encode4Input = torch.cat([encode1DownSampled2, encode2DownSampled1, encode3], dim=1)
		encode4Input = self.combine2(encode4Input)

		encode4, skip4 = self.encode4(resizedInputLayer3, encode4Input)   

		encode1DownSampled3 = self.pooling(encode1DownSampled2)
		encode2DownSampled2 = self.pooling(encode2DownSampled1)
		encode3DownSampled1 = self.pooling(encode3)
		output = torch.cat([encode1DownSampled3, encode2DownSampled2, encode3DownSampled1, encode4], dim=1)
		output = self.combine3(output)

		return output, skip1, skip2, skip3, skip4
	
class EncoderBlockSPADEWideStart(torch.nn.Module):
	def __init__(self, inFilterCount:int, outfilterCount:int):
		super().__init__()

		self.wide = WideStart(inFilterCount, outfilterCount)
		self.spade = DenseSPADEConvBlock(outfilterCount, outfilterCount)
		
		self.shrink = torch.nn.Sequential(
			torch.nn.ReplicationPad2d((0,1,0,1)),
			torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
			)

	def forward(self, inputTensor:torch.Tensor):
		wideStart = self.wide(inputTensor)
		commonSkip = self.spade(inputTensor, wideStart)

		avgPool = self.shrink(commonSkip)
		return avgPool, commonSkip
	
class WideStart(torch.nn.Module):
	def __init__(self, inFilterCount:int, outfilterCount:int):
		super().__init__()
		groupOutCount = inFilterCount*32
		self.wideKernelH = torch.nn.Conv2d(inFilterCount, groupOutCount, kernel_size=(9,1), groups=inFilterCount)
		self.wideKernelV = torch.nn.Conv2d(inFilterCount, groupOutCount, kernel_size=(1,9), groups=inFilterCount)
		self.centered = DepthWiseConvWithPadding(inFilterCount, groupOutCount, kernel_size=3)

		self.mixer = torch.nn.Conv2d(groupOutCount*3, outfilterCount, kernel_size=1)

		self.norm1 = ChannelFirstLayerNorm(outfilterCount)

	def forward(self, inputLayer):
		padV = torch.nn.functional.pad(inputLayer, (4,4,0,0), mode='reflect')
		padH = torch.nn.functional.pad(inputLayer, (0,0,4,4), mode='reflect')
		wide1 = self.wideKernelH(padH)
		wide2 = self.wideKernelV(padV)
		center = self.centered(inputLayer)
		cat = torch.cat([wide1, wide2, center], dim=1)
		mixed = self.mixer(cat)
		normed = self.norm1(mixed)
		activated = torch.nn.functional.gelu(normed)
		return activated
	
class DenseSPADEConvBlock(torch.nn.Module):
	def __init__(self, channelsIn, filterCount, kernel_size=5, padmode='replicate'):
		super().__init__()
		
		self.channelCountOut = filterCount
		self.kernel_size = kernel_size
		p = self.padding_required = kernel_size // 2
		self.paddingTuple = (p,p,p,p)
		self.pad_mode = padmode
		if padmode == 'replicate':
			self.padder = torch.nn.ReflectionPad2d(self.paddingTuple)
		elif padmode == 'reflect':
			self.padder = torch.nn.ReflectionPad2d(self.paddingTuple)

		self.layerNorm = ChannelFirstLayerNorm(filterCount)
		self.activation = torch.nn.GELU()

		self.conv1 = DepthWiseConv(channelsIn, filterCount, kernel_size=kernel_size, stride=1)
		self.spade1 = SPADEBlock(filterCount)

		self.conv2 = DepthWiseConv(filterCount, filterCount, kernel_size=kernel_size, stride=1)
		self.spade2 = SPADEBlock(filterCount)

		self.combine1 = torch.nn.Conv2d(filterCount * 2, filterCount, kernel_size=1)

		self.conv3 = DepthWiseConv(filterCount, filterCount, kernel_size=kernel_size, stride=1)
		self.spade3 = SPADEBlock(filterCount)

		self.combine2 = torch.nn.Conv2d(filterCount * 3, filterCount, kernel_size=1)

	def forward(self, rgbTensor:torch.Tensor, inputTensor:torch.Tensor):
		#DW conv
		padded1 = self.padder(inputTensor)
		conv1 = self.conv1(padded1)

		#SPADE
		spade1 = self.spade1(rgbTensor, conv1)

		#GeLU
		activated1 = self.activation(spade1)

		#DW conv
		padded2 = self.padder(activated1)
		conv2 = self.conv2(padded2)

		combined1 = self.combine1(torch.cat((conv1, conv2), dim=1))

		#SPADE
		spade2 = self.spade2(rgbTensor, combined1)

		#GeLU
		activated2 = self.activation(spade2)

		#DW conv
		padded3 = self.padder(activated2)
		conv3 = self.conv3(padded3)

		combined2 = self.combine2(torch.cat((conv1, conv2, conv3), dim=1))

		#SPADE
		spade3 = self.spade3(rgbTensor, combined2)

		#GeLU
		activated3 = self.activation(spade3)
		
		return activated3
	
class EncoderBlockSPADE(torch.nn.Module):
	def __init__(self, inFilterCount:int, outfilterCount:int):
		super().__init__()

		self.common = DenseSPADEConvBlock(inFilterCount, outfilterCount)
		self.shrink = torch.nn.Sequential(
			torch.nn.ReplicationPad2d((0,1,0,1)),
			torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
			)

	def forward(self, rgbTensor:torch.Tensor, inputTensor:torch.Tensor):
		commonSkip = self.common(rgbTensor, inputTensor)
		avgPool = self.shrink(commonSkip)
		return avgPool, commonSkip
	
class ChannelFirstLayerNorm(torch.nn.Module):
	def __init__(self, normalized_shape, eps=1e-6):
		super().__init__()
		self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
		self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
		self.eps = eps
	
	def forward(self, x):
		u = x.mean(1, keepdim=True)
		s = (x - u).pow(2).mean(1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.eps)
		x = self.weight[:, None, None] * x + self.bias[:, None, None]
		return x
	
class DepthWiseConv(torch.nn.Module):
	def __init__(self, channelsIn, channelsOut, kernel_size, stride=1):
		super().__init__()
		self.stride = stride
		self.kernelSize = kernel_size
		self.channelsIn = channelsIn
		self.channelsOut = channelsOut
		self.depthWise = torch.nn.Conv2d(channelsIn, channelsIn, kernel_size=kernel_size, stride=stride, groups=channelsIn)
		self.conv1x1 = torch.nn.Conv2d(channelsIn, channelsOut, kernel_size=1, stride=1, groups=1)
		self.InitWeights(self.depthWise, self.channelsIn)
		self.InitWeights(self.conv1x1, self.channelsOut)

	def InitWeights(self, layer, channels):
		gstd = WeightInitStd(3, channels)

		if type(layer) == torch.nn.Conv2d:
			torch.nn.init.normal_(layer.weight, mean=0.0, std=gstd)
			layer.bias.data.fill_(0.0)

	def forward(self, inputLayer):
		depthWiseTensor = self.depthWise(inputLayer) # one filter per input channel.
		pieceWise = self.conv1x1(depthWiseTensor)  # 1x1 filters across all inputs to get outputs.
		return pieceWise
	
def WeightInitStd(kernelsize, channels):
	# example given: 3x3 and 64 channels,  3*3*64 = 576
	n = kernelsize * kernelsize * channels
	gaussian_std = math.sqrt(2.0 / n)
	return gaussian_std

class SPADEBlock(torch.nn.Module):
	def __init__(self, inChannels:int):
		super().__init__()
		
		self.inChannels = inChannels

		self.layerNorm = ChannelFirstLayerNorm(inChannels)

		self.rgbEmbeddingConv = torch.nn.Conv2d(3, 128, kernel_size=3)

		self.attentionMaskConv = torch.nn.Conv2d(128, self.inChannels, kernel_size=3)
		self.additiveConnectionConv = torch.nn.Conv2d(128, self.inChannels, kernel_size=3)

		self.padding = torch.nn.ReflectionPad2d((1, 1, 1, 1))

	def forward(self, rgbTensor:torch.Tensor, inputTensor:torch.Tensor):
		paddedRGB = self.padding(rgbTensor)

		#layer norm on input tensor
		normed = self.layerNorm(inputTensor)

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
	
class DepthWiseConvWithPadding(DepthWiseConv):
	def __init__(self, channelsIn, channelsOut, kernel_size, stride=1, padding=1):
		super().__init__(channelsIn, channelsOut, kernel_size, stride)
		self.padding = (padding,padding,padding,padding)

	def forward(self, inputLayer):
		padded = torch.nn.functional.pad(inputLayer, self.padding, mode='replicate')
		depthWiseTensor = self.depthWise(padded) # one filter per input channel.
		pieceWise = self.conv1x1(depthWiseTensor)  # 1x1 filters across all inputs to get outputs.
		return pieceWise