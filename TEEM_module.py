import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class TEEM(nn.Module):

    def __init__(self,chn,final):
        super(TEEM,self).__init__()
        self.channel = chn
        self.dual_channel = chn*2
        self.inter_mid = int(chn/2)
        self.final_channel = final
        if self.final_channel == 0:
            self.final_channel = 2
        self.early_conv2d_1  = conv3x3(self.dual_channel, 1, stride=1)
        self.early_conv2d_3  = conv3x3(self.channel, self.final_channel, stride=1)
        #self.relu1 = nn.Sigmoid()
        self.relu2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(self.final_channel, 2)
        self.norm1 = nn.BatchNorm2d( self.inter_mid)
        self.norm2 = nn.BatchNorm2d(self.final_channel)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_1, input_2):
        #difference of feature maps
        difference = input_2 - input_1
        x = torch.cat((difference,input_2),1) 
        x = self.early_conv2d_1(x)
        x = torch.sigmoid(x)
        x = x.expand_as(input_2)
        x = x*input_2
        x = self.early_conv2d_3(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.avg(x)
        output  = self.classifier(torch.flatten(x, 1))
        output = self.softmax(output)
        return output


class ChangeDetectorDoubleAttDyn4(nn.Module):

    def __init__(self,chn,final):
        super().__init__()
        self.channel = chn
        self.dual_channel = chn*2
        self.inter_mid = int(chn/2)
        self.final_channel = final
        if self.final_channel == 0:
            self.final_channel = 2
        self.early_conv2d_1  = conv3x3(self.dual_channel, 1, stride=1)
        self.early_conv2d_3  = conv3x3(self.channel, self.final_channel, stride=1)
        #self.relu1 = nn.Sigmoid()
        self.relu2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(self.final_channel, 2)
        self.norm1 = nn.BatchNorm2d( self.inter_mid)
        self.norm2 = nn.BatchNorm2d(self.final_channel)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_1, input_2):
        #difference of feature maps
        difference = input_2 - input_1
        x = torch.cat((difference,input_2),1) 
        x = self.early_conv2d_1(x)
        x = torch.sigmoid(x)
        x = x.expand_as(input_2)
        x = x*input_2
        x = self.avg(x)
        output  = self.classifier(torch.flatten(x, 1))
        output = self.softmax(output)
        return output
class ChangeDetectorDoubleAttDyn5(nn.Module):

    def __init__(self,chn,final):
        super().__init__()
        self.channel = chn
        self.dual_channel = chn*2
        self.inter_mid = int(chn/2)
        self.final_channel = final
        if self.final_channel == 0:
            self.final_channel = 2
        self.early_conv2d_1  = conv3x3(self.dual_channel, 1, stride=1)
        self.early_conv2d_3  = conv3x3(self.channel, self.final_channel, stride=1)
        #self.relu1 = nn.Sigmoid()
        self.relu2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(self.final_channel, 2)
        self.norm1 = nn.BatchNorm2d( self.inter_mid)
        self.norm2 = nn.BatchNorm2d(self.final_channel)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_1, input_2):
        #difference of feature maps
        difference = input_2 - input_1
        #x = torch.cat((difference,input_2),1) 
        #x = self.early_conv2d_1(x)
        #x = torch.sigmoid(x)
        #x = x.expand_as(input_2)
        x = difference*input_2
        x = self.avg(x)
        output  = self.classifier(torch.flatten(x, 1))
        output = self.softmax(output)
        return output
class ChangeDetectorDoubleAttDyn2(nn.Module):

    def __init__(self,chn,middle,final):
        super().__init__()
        self.channel = chn
        self.dual_channel = chn*2
        self.inter_mid = middle
        self.final_channel = final
        if self.final_channel == 0:
            self.final_channel = 2
        self.early_conv2d_1  = conv3x3(self.dual_channel, 1, stride=1)
        self.early_conv2d_3  = conv3x3(self.channel, self.inter_mid, stride=1)
        self.early_conv2d_4  = conv3x3(self.inter_mid, self.final_channel, stride=1)
        self.max_pooling = torch.nn.MaxPool2d(3,stride=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(self.final_channel, 2)
        self.norm1 = nn.BatchNorm2d( self.inter_mid)
        self.norm2 = nn.BatchNorm2d(self.final_channel)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_1, input_2):
        #difference of feature maps
        difference = input_2 - input_1
        x = torch.cat((difference,input_2),1) 
        x = self.early_conv2d_1(x)
        x = torch.sigmoid(x)
        x = x.expand_as(input_2)
        x = x*input_2
        x = self.early_conv2d_3(x)
        x = self.norm1(x)
        x = self.relu2(x)
        x = self.max_pooling(x)
        x = self.early_conv2d_4(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.avg(x)#torch.sum(x,(2,3))
        output  = self.classifier(torch.flatten(x, 1))
        output = self.softmax(output)
        return output

class ChangeDetectorDoubleAttDyn_orig(nn.Module):

    def __init__(self,chn):
        super().__init__()
        self.channel = chn
        self.dual_channel = chn*2
        self.final_channel = int(chn/2)
        self.early_conv2d_1  = conv3x3(self.dual_channel, self.channel, stride=1)
        self.early_conv2d_3  = conv3x3(self.channel, self.final_channel, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(self.final_channel, 2)
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.final_channel)

    def forward(self, input_1, input_2):
        #difference of feature maps
        difference = input_2 - input_1
        x = torch.cat((difference,input_2),1) 
        x = self.early_conv2d_1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = x*input_2
        x = self.early_conv2d_3(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = torch.sum(x,(2,3))
        output  = self.classifier(torch.flatten(x, 1))

        return output

class ChangeDetectorDoubleAttDyn3(nn.Module):

    def __init__(self,chn,final):
        super().__init__()
        self.channel = chn
        self.dual_channel = chn*2
        self.inter_mid = int(chn/2)
        self.final_channel = final
        if self.final_channel == 0:
            self.final_channel = 2
        self.early_conv2d_1  = conv1x1(self.dual_channel, 1, stride=1)
        self.early_conv2d_2  = conv1x1(self.inter_mid, 1, stride=1)
        #self.early_conv2d_1  = conv3x3(self.dual_channel, 1, stride=1)
        self.early_conv2d_3  = conv1x1(self.channel, self.final_channel, stride=1)
        #self.relu1 = nn.Sigmoid()
        self.relu2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(self.final_channel, 2)
        self.norm1 = nn.BatchNorm2d( self.inter_mid)
        self.norm2 = nn.BatchNorm2d(self.final_channel)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_1, input_2):
        #difference of feature maps
        difference = input_2 - input_1
        x = torch.cat((difference,input_2),1) 
        x = self.early_conv2d_1(x)
        x = torch.sigmoid(x)
        x = x.expand_as(input_2)
        x = x*input_2
        x = self.early_conv2d_3(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.avg(x)
        output  = self.classifier(torch.flatten(x, 1))
        output = self.softmax(output)
        return output
