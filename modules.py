import torch
from torch import nn as nn
import torch.nn.functional as F


class BatchNormalization2D(nn.Module):
    def __init__(self, numFeatures):
        super(BatchNormalization2D, self).__init__()
        self.layer = nn.BatchNorm2d(num_features=numFeatures, momentum=1-0.99)

    def forward(self, x):
        return self.layer(x)#.clamp(1e-3)

class PointWiseConvolution(nn.Module):
    def __init__(self, inChannels, outChannels, stride, expansionFactor, isNormal):
        super(PointWiseConvolution, self).__init__()
        if isNormal:
            self.layer = nn.Conv2d(in_channels=inChannels*expansionFactor,out_channels=outChannels, kernel_size=1, stride=stride, bias=True)
        else:
            self.layer = nn.Conv2d(in_channels=inChannels,out_channels=inChannels*expansionFactor, kernel_size=1, stride=stride, bias=True)
        
    def forward(self, x):
        return self.layer(x)#.clamp(1e-3))


class DepthWiseConvolution(nn.Module):
    def __init__(self, channels, kernelSize, stride, expansionFactor):
        super(DepthWiseConvolution, self).__init__()
        channels = channels*expansionFactor
        self.layer = nn.Conv2d(channels, channels, kernelSize, stride, (kernelSize-1)//2,groups=channels, bias=True)

    def forward(self, x):
        return self.layer(x)#.clamp(1e-3))

class MultiKernelDepthWiseConvolution(nn.Module):
    def __init__(self, channels, expansionFactor, nChunks=1, stride=1, bias=True):
        super(MultiKernelDepthWiseConvolution, self).__init__()
        self.nChunks = nChunks
        #self.splitOutChannels = u.split_layer(outChannels, nChunks)
        self.layers = nn.ModuleList()
        self.channels = channels
        self.outChannels = (channels*expansionFactor)//nChunks
        """
        for i in range(self.nChunks):
            kernelSize = 2 * i + 3
            self.layers.append(DepthWiseConvolution(self.splitOutChannels[i], kernelSize, stride, expansionFactor))
        """
        #self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 1, stride, 1),
        # BatchNormalization2D((channels*expansionFactor)//nChunks),
        # MemoryEfficientSwish()))
        self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 3, stride, 1), SqueezeAndExcitation(self.outChannels, self.outChannels//4), BatchNormalization2D(self.outChannels), MemoryEfficientSwish(),  BatchNormalization2D((channels*expansionFactor)//nChunks),MemoryEfficientSwish()))
        self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 5, stride, 1), SqueezeAndExcitation(self.outChannels, self.outChannels//4), BatchNormalization2D(self.outChannels), MemoryEfficientSwish(),  BatchNormalization2D((channels*expansionFactor)//nChunks), MemoryEfficientSwish()))
        self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 7, stride, 1), SqueezeAndExcitation(self.outChannels, self.outChannels//4), BatchNormalization2D(self.outChannels), MemoryEfficientSwish()))
        #self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 7, stride, 1)))#, BatchNormalization2D(self.outChannels), MemoryEfficientSwish()))
        #self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 7, stride, 1),  BatchNormalization2D((channels*expansionFactor)//nChunks), MemoryEfficientSwish()))
        self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 9, stride, 1), SqueezeAndExcitation(self.outChannels, self.outChannels//4), BatchNormalization2D((channels*expansionFactor)//nChunks), MemoryEfficientSwish()))
        #self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 11, stride, 1)))#,  BatchNormalization2D((channels*expansionFactor)//nChunks), MemoryEfficientSwish()))

        #self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 2, 1, 1), MemoryEfficientSwish()))
        #self.layers.append(nn.Sequential(DepthWiseConvolution((channels*expansionFactor)//nChunks, 2, 1, 1), MemoryEfficientSwish()))

    def forward(self, x):
        #print(x.size())
        split = torch.split(x, self.outChannels, dim=1)
        #print(split[0].size())
        x = torch.cat([layer(s) for layer, s in zip(self.layers, split)], dim=1)
        #x = self.bn(x)
        #print(x.size(0))
        return x

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reductionDimension):
        super(SqueezeAndExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reductionDimension, 1),
            MemoryEfficientSwish(),
            nn.Conv2d(reductionDimension, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return (y * x)#.clamp(1e-3)


class MBConv(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, dropoutProbability, expansionFactor):
        super(MBConv, self).__init__()

        self.skip = inChannels == outChannels and stride == 1
        reductionDimension = inChannels//4 if inChannels > 4 else 1 

        if expansionFactor != 1:
            self.expansionLayer = nn.Sequential(
                PointWiseConvolution(inChannels, outChannels, stride, expansionFactor, False),
                BatchNormalization2D(inChannels*expansionFactor),
                MemoryEfficientSwish())
        else:
            self.expansionLayer = nn.Identity()

        self.depthWiseLayer =  MultiKernelDepthWiseConvolution(inChannels, expansionFactor, 4, stride) if not self.skip else DepthWiseConvolution(inChannels,kernelSize,stride, expansionFactor)
        self.b = BatchNormalization2D(inChannels*expansionFactor)
        self.sw = MemoryEfficientSwish()
        self.squeezeAndExcitationLayer = SqueezeAndExcitation(inChannels*expansionFactor, reductionDimension)

        self.reductionLayer = nn.Sequential(
            PointWiseConvolution(inChannels, outChannels, stride, expansionFactor, True),
            BatchNormalization2D(outChannels),
            MemoryEfficientSwish())

        self.dropoutLayer = nn.Dropout2d(dropoutProbability)

    def forward(self, x):
        y = x
        #print(x.size(0))
        y = self.expansionLayer(y)
        y = self.depthWiseLayer(y)
        y = self.b(y)
        y = self.sw(y)
        y = self.squeezeAndExcitationLayer(y)
        y = self.reductionLayer(y)
        if self.skip:
            y = self.dropoutLayer(y)
            y = y + x
        return y
    
class EfficientNet(nn.Module):
    def __init__(self, networkParametersDict, dropoutProbability):
        super(EfficientNet, self).__init__()

        self.stages = nn.Sequential()
        for _, (key, config) in enumerate(networkParametersDict.items()):
            if key == "Top":
                self.stages.add_module("Top: Conv2d",nn.Conv2d(config["inChannels"], config["outChannels"], config["kernelSize"], 1))
                self.stages.add_module("Top: Batch Normalization", BatchNormalization2D(config["outChannels"]))
                self.stages.add_module("Top: Linearility", MemoryEfficientSwish())
                
            elif key == "Stage 1":
                self.stages.add_module("{} : Conv2d".format(key), nn.Conv2d(config["inChannels"], config["outChannels"], config["kernelSize"], 2))
                self.stages.add_module("{} : Batch Normalization".format(key), BatchNormalization2D(config["outChannels"]))
                self.stages.add_module("{} : Linearility".format(key), MemoryEfficientSwish())

            elif key == "Final":
                self.final = nn.ModuleList([
                    #nn.Conv2dSame(config["inChannels"], config["outChannels"], config["kernelSize"], 2),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Dropout(dropoutProbability),
                    nn.Linear(config["inChannels"], config["outChannels"]),
                    MemoryEfficientSwish()
                ])
            else:
                for i in range(config["repeats"]):
                    if i == config["repeats"] - 1:
                        self.stages.add_module("{}: Layer {} MBConv{}".format(key, i, config["expansionFactor"]), MBConv(config["inChannels"], config["outChannels"], config["kernelSize"], 1, dropoutProbability, config["expansionFactor"]))
                    else:
                        self.stages.add_module("{}: Layer {} MBConv{}".format(key, i, config["expansionFactor"]), MBConv(config["inChannels"], config["inChannels"], config["kernelSize"], 2 if i == 0 else 1, dropoutProbability, config["expansionFactor"]))
    
    def forward(self, x):
        x = self.stages(x)
        x = self.final[0](x)
        x = x.flatten(start_dim=1)
        x = self.final[1](x)
        x = self.final[2](x)
        x = self.final[3](x)
        return x

#credit to lukemas
# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
        

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Net(nn.Module):
    def __init__(self, nc, dp=0.2):
        super(Net, self).__init__()
        self.init_batch_norm = BatchNormalization2D(3)
        self.head = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3, stride=1)
        self.swish = MemoryEfficientSwish()
        self.bn = BatchNormalization2D(24)
        self.config = [
            (24, 48, 1, 2, 1),
            (48, 64, 2, 1, 3),
            (64, 128, 1, 2, 3),
            (128, 256, 4, 1, 1)
            #(128, 160, 9, 1, 6),
            #(160, 272, 15, 2, 6),
            ]
        self.stages = nn.ModuleList([nn.Sequential() for stage in self.config])
        for i in range(len(self.config)):
            for j in range(self.config[i][2]):
                self.stages[i].add_module(str(j+1), MBConv(self.config[i][0], self.config[i][0], 3, 1, 0, self.config[i][4]) if j is not self.config[i][2] - 1 else MBConv(self.config[i][0], self.config[i][1], 1, self.config[i][3], 0, self.config[i][4]))
        self.final_conv = nn.Sequential(nn.Conv2d(self.config[-1][1], 512, 1, 1), BatchNormalization2D(512), MemoryEfficientSwish())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, nc)
    def forward(self, x):
        x = self.head(x)
        x = self.bn(x)
        x = self.swish(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_conv(x)
        x = self.gap(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


