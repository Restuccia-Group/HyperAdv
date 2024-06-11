import numpy as np
import torch
from torch import nn
import math    

"""
Basic 1D-CNN for wireless tasks
"""

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel,out_channels,3,padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.conv1(x)
        return nn.MaxPool1d(2)(x)
    
class ConvNet(nn.Module):
    def __init__(self,in_channel,nclasses):
        super(ConvNet, self).__init__()

        self.backbone = nn.ModuleList()
        self.backbone.append(ConvBlock(in_channel,64))
        self.backbone.append(ConvBlock(64,64))
        self.backbone.append(ConvBlock(64,128)) # 128
        self.backbone.append(ConvBlock(128,128)) # 64
        self.backbone.append(ConvBlock(128,256)) # 32
        self.backbone.append(ConvBlock(256,256)) # 16
        self.exit = nn.Sequential(
            nn.Linear(256,nclasses)
        )
    def forward(self,x):
        for i,e in enumerate(self.backbone):
            x = e(x)
        x = torch.mean(x,dim=2)
        x = self.exit(x)
        return x
    
"""
HyperNetwork: 
    a framework to learn to generate deep neural network
    context: some predifined context for tasks
    hyper: hyper network which will take context as input and generate weight for target network
    target: target network used for inference

┌───────┐      ┌────────┐ 
│context│ ───> |  Hyper | 
└───────┘      └────┬───┘ 
                    |     
┌───────┐      ┌────┴───┐ 
│ input │ ───> | Target | 
└───────┘      └────┬───┘ 
                    |
               ┌────┴───┐
               | Output |
               └────────┘
"""

class HyperConv(nn.Module):
    """
    Conv Hyper: hypernet to take context and generate parameters for a conv layer
        arguments:
            context_dim: dimension of the context vector
            hidden_dim: dimension of the hidden layer of hyper network
            conv_channel_in: channel dimension of conv input
            conv_channel_out: channel dimension of conv output
        input:
            context: a batch of context vectors (batchsize, context_dim)
        output:
            conv_weight: a batch of weight for conv layer (batchsize, conv_channel_out, conv_channel_in, 3)
            conv_bias: a batch of bias for conv layer (batchsize, conv_channel_out)
    """
    def __init__(self, context_dim, hidden_dim, conv_channel_in, conv_channel_out):
        super(HyperConv, self).__init__()
        self.hyper_conv_w = nn.Sequential(
            nn.Linear(context_dim,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,conv_channel_out*conv_channel_in*3),
        )
        self.hyper_conv_b = nn.Sequential(
            nn.Linear(context_dim,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,conv_channel_out),
        )
        self.channel_in = conv_channel_in
        self.channel_out = conv_channel_out
    def forward(self,context):
        conv_weight = self.hyper_conv_w(context).reshape(-1,self.channel_out,self.channel_in,3)
        conv_bias = self.hyper_conv_b(context)
        return conv_weight, conv_bias
    
class ConvTargetBlock(nn.Module):
    """
    Conv Target: target conv layer to process input x
        input:
            x: a batch of input
            weights: a tuple of (
                     conv weight (out_channel, in_channel, 3), 
                     conv bias (out_channel) )
        output:
            a batch of processed features
    """
    def __init__(self):
        super(ConvTargetBlock, self).__init__()

    def forward(self, x, conv_w, conv_b):
        x = nn.functional.conv1d(x,conv_w,bias=conv_b,padding='same')
        x = nn.ReLU()(x)
        return nn.MaxPool1d(2)(x)
    
class LinearTargetBlock(nn.Module):
    """
    Linear Target: target linear layer
        input:
            x: a batch of input (batch, dim_in)
            weights: weight (dim_out,dim_in) and bias (dim_out) of the target layer
    """
    def __init__(self):
        super(LinearTargetBlock, self).__init__()
    def forward(self,x,w,b):
        x = nn.functional.linear(x,w,b)
        return x
    
class HyperLinear(nn.Module):
    """
    Linear Hyper: hypernet to take context and generate parameters for a linear layer
        arguments:
            context_dim: dimension of the context vector
            hidden_dim: hidden dimension of the hyper network
            linear_in: input dimension of the target linear layer
            linear_out: output dimension of the target linear layer
        input:
            context: a batch of context (batch, context_dim)
        output:
            weights: a batch of weights of target linear layer (batch, linear_out, linear_in)
            bias: a batch of bias of target linear layer (batch, linear_out)
    """
    def __init__(self, context_dim, hidden_dim, linear_in, linear_out):
        super(HyperLinear, self).__init__()
        self.dim_in = linear_in
        self.dim_out = linear_out
        self.hyper_w = nn.Sequential(
            nn.Linear(context_dim,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,linear_in*linear_out),
        )
        self.hyper_b = nn.Sequential(
            nn.Linear(context_dim,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,linear_out),
        )
    def forward(self,context):
        weights = self.hyper_w(context).reshape(-1,self.dim_out,self.dim_in)
        bias = self.hyper_b(context)
        return weights,bias
    
class HyperCNN(nn.Module):
    def __init__(self,in_channel,nclasses,h_dim,ntasks,weight=None,bias=None):
        super(HyperCNN, self).__init__()
        if weight is None:
            weight = torch.rand((ntasks,nclasses))*2 - 1
        if bias is None:
            bias = torch.rand((ntasks,nclasses))*2 - 1
        self.register_buffer('weight',weight)
        self.register_buffer('bias',bias)
        self.conv1_hyper = HyperConv(nclasses*2,h_dim,in_channel,64)
        self.conv2_hyper = HyperConv(nclasses*2,h_dim,64,64)
        self.conv3_hyper = HyperConv(nclasses*2,h_dim,64,128)
        self.conv4_hyper = HyperConv(nclasses*2,h_dim,128,128)
        self.conv5_hyper = HyperConv(nclasses*2,h_dim,128,256)
        self.conv6_hyper = HyperConv(nclasses*2,h_dim,256,256)
        self.exit_hyper = HyperLinear(nclasses*2,h_dim,256,nclasses)

        self.conv1_target = ConvTargetBlock()
        self.conv2_target = ConvTargetBlock()
        self.conv3_target = ConvTargetBlock()
        self.conv4_target = ConvTargetBlock()
        self.conv5_target = ConvTargetBlock()
        self.conv6_target = ConvTargetBlock()
        self.exit_target = LinearTargetBlock()

    def forward(self,input,training=True,ensemble=False,teach=False):

        if training or ensemble:
            context = torch.cat((self.weight,self.bias),1)
        else:
            idx = np.random.randint(self.weight.shape[0])
            context = torch.cat((self.weight[idx].reshape(1,-1),self.bias[idx].reshape(1,-1)),1)
        conv1_w, conv1_b = self.conv1_hyper(context)
        conv2_w, conv2_b = self.conv2_hyper(context)
        conv3_w, conv3_b = self.conv3_hyper(context)
        conv4_w, conv4_b = self.conv4_hyper(context)
        conv5_w, conv5_b = self.conv5_hyper(context)
        conv6_w, conv6_b = self.conv6_hyper(context)
        exit_w, exit_b = self.exit_hyper(context)
        if teach:
            return conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, conv4_w, conv4_b, conv5_w, conv5_b, conv6_w, conv6_b, exit_w, exit_b
        logits = []
        if training or ensemble:
            for i in range(context.shape[0]):
                x = self.conv1_target(input, conv1_w[i], conv1_b[i])
                x = self.conv2_target(x, conv2_w[i], conv2_b[i])
                x = self.conv3_target(x, conv3_w[i], conv3_b[i])
                x = self.conv4_target(x, conv4_w[i], conv4_b[i])
                x = self.conv5_target(x, conv5_w[i], conv5_b[i])
                x = self.conv6_target(x, conv6_w[i], conv6_b[i])
                x = torch.mean(x,dim=2)
                x = self.exit_target(x,exit_w[i],exit_b[i])
                x = self.weight[i].reshape(1,-1)*x+self.bias[i].reshape(1,-1) # calibrate the logits with context
                logits.append(x)
        else:
            x = self.conv1_target(input, conv1_w[0], conv1_b[0])
            x = self.conv2_target(x, conv2_w[0], conv2_b[0])
            x = self.conv3_target(x, conv3_w[0], conv3_b[0])
            x = self.conv4_target(x, conv4_w[0], conv4_b[0])
            x = self.conv5_target(x, conv5_w[0], conv5_b[0])
            x = self.conv6_target(x, conv6_w[0], conv6_b[0])
            x = torch.mean(x,dim=2)
            x = self.exit_target(x,exit_w[0],exit_b[0])
            x = self.weight[idx].reshape(1,-1)*x+self.bias[idx].reshape(1,-1) # calibrate the logits with context
            logits = x

        if ensemble:
            ensemble_logits = torch.zeros_like(logits[0])
            for logit in logits:
                ensemble_logits += logit
            ensemble_logits /= len(logits)
            logits = ensemble_logits

        return logits
    
class ChunkwiseLinear(nn.Module):
    """
    Chunkwise Linear layer: customize linear layer to reduce mdoel complexity, similar to group convolution
        arguments:
            in_dim: dimension of input, must be divisible by nchunk
            out_dim: dimension of output, must be divisible by nchunk
            nchunk: number of identical chunks 
        input:
            x: batch of input tensor, whose shape must be batchsize x in_dim
        output:
            out: output tensor whose shape is batchsize x out_dim
    """
    def __init__(self,in_dim,out_dim,nchunk):
        super(ChunkwiseLinear,self).__init__()
        assert in_dim%nchunk == 0
        assert out_dim%nchunk == 0
        self.weights = nn.Parameter(torch.FloatTensor(out_dim,in_dim//nchunk).uniform_(-math.sqrt(1/in_dim),math.sqrt(1/in_dim)))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim).uniform_(-math.sqrt(1/in_dim),math.sqrt(1/in_dim)))
        self.nchunk = nchunk
    def forward(self,x):
        out_ls = []
        x_ls = torch.tensor_split(x,self.nchunk,dim=1)
        w_ls = torch.tensor_split(self.weights,self.nchunk,dim=0)
        b_ls = torch.tensor_split(self.bias,self.nchunk,dim=0)
        for i in range(self.nchunk):
            out_ls.append(nn.functional.linear(x_ls[i],w_ls[i],b_ls[i]))
        out = torch.cat(out_ls,dim=1)
        return out
    
class HyperConvChunk(nn.Module):
    """
    Chunkwise Conv Hyper: Replace the second linear layer in ConvHyper with chunkwise linear
        arguments:
            context_dim: dimension of the context vector
            hidden_dim: dimension of the hidden layer of hyper network
            conv_channel_in: channel dimension of conv input
            conv_channel_out: channel dimension of conv output
            nchunk: number of chunks
        input:
            context: a batch of context vectors (batchsize, context_dim)
        output:
            conv_weight: a batch of weight for conv layer (batchsize, conv_channel_out, conv_channel_in, 3)
            conv_bias: a batch of bias for conv layer (batchsize, conv_channel_out)
    """
    def __init__(self, context_dim, hidden_dim, conv_channel_in, conv_channel_out,nchunk=8):
        super(HyperConvChunk, self).__init__()
        self.hyper_conv_w = nn.Sequential(
            nn.Linear(context_dim,hidden_dim),
            nn.ReLU(inplace=True),
            ChunkwiseLinear(hidden_dim,conv_channel_out*conv_channel_in*3,nchunk),
        )
        self.hyper_conv_b = nn.Sequential(
            nn.Linear(context_dim,hidden_dim),
            nn.ReLU(inplace=True),
            ChunkwiseLinear(hidden_dim,conv_channel_out,nchunk),
        )
        self.channel_in = conv_channel_in
        self.channel_out = conv_channel_out
    def forward(self,context):
        conv_weight = self.hyper_conv_w(context).reshape(-1,self.channel_out,self.channel_in,3)
        conv_bias = self.hyper_conv_b(context)
        return conv_weight, conv_bias
    
class HyperLinearChunk(nn.Module):
    """
    Chunkwise Linear Hyper: Replace the second linear layer in LinearHyper with chunkwise linear
        arguments:
            context_dim: dimension of the context vector
            hidden_dim: hidden dimension of the hyper network
            linear_in: input dimension of the target linear layer
            linear_out: output dimension of the target linear layer
            nchunk: number of chunks
        input:
            context: a batch of context (batch, context_dim)
        output:
            weights: a batch of weights of target linear layer (batch, linear_out, linear_in)
            bias: a batch of bias of target linear layer (batch, linear_out)
    """
    def __init__(self, context_dim, hidden_dim, linear_in, linear_out, nchunk=8):
        super(HyperLinearChunk, self).__init__()
        self.dim_in = linear_in
        self.dim_out = linear_out
        self.hyper_w = nn.Sequential(
            nn.Linear(context_dim,hidden_dim),
            nn.ReLU(inplace=True),
            ChunkwiseLinear(hidden_dim,linear_in*linear_out,nchunk),
        )
        self.hyper_b = nn.Sequential(
            nn.Linear(context_dim,hidden_dim),
            nn.ReLU(inplace=True),
            ChunkwiseLinear(hidden_dim,linear_out,nchunk),
        )
    def forward(self,context):
        weights = self.hyper_w(context).reshape(-1,self.dim_out,self.dim_in)
        bias = self.hyper_b(context)
        return weights,bias
    
class HyperCNNChunk(nn.Module):
    def __init__(self,in_channel,nclasses,h_dim,ntasks,weight=None,bias=None):
        super(HyperCNNChunk, self).__init__()
        if weight is None:
            weight = torch.rand((ntasks,nclasses))*2 - 1
        if bias is None:
            bias = torch.rand((ntasks,nclasses))*2 - 1
        self.register_buffer('weight',weight)
        self.register_buffer('bias',bias)
        self.conv1_hyper = HyperConvChunk(nclasses*2,h_dim,in_channel,64)
        self.conv2_hyper = HyperConvChunk(nclasses*2,h_dim,64,64)
        self.conv3_hyper = HyperConvChunk(nclasses*2,h_dim,64,128)
        self.conv4_hyper = HyperConvChunk(nclasses*2,h_dim,128,128)
        self.conv5_hyper = HyperConvChunk(nclasses*2,h_dim,128,256)
        self.conv6_hyper = HyperConvChunk(nclasses*2,h_dim,256,256)
        self.exit_hyper = HyperLinearChunk(nclasses*2,h_dim,256,nclasses)

        self.conv1_target = ConvTargetBlock()
        self.conv2_target = ConvTargetBlock()
        self.conv3_target = ConvTargetBlock()
        self.conv4_target = ConvTargetBlock()
        self.conv5_target = ConvTargetBlock()
        self.conv6_target = ConvTargetBlock()
        self.exit_target = LinearTargetBlock()

    def forward(self,input,training=True,ensemble=False,teach=False):

        if training or ensemble:
            context = torch.cat((self.weight,self.bias),1)
        else:
            idx = np.random.randint(self.weight.shape[0])
            context = torch.cat((self.weight[idx].reshape(1,-1),self.bias[idx].reshape(1,-1)),1)
        conv1_w, conv1_b = self.conv1_hyper(context)
        conv2_w, conv2_b = self.conv2_hyper(context)
        conv3_w, conv3_b = self.conv3_hyper(context)
        conv4_w, conv4_b = self.conv4_hyper(context)
        conv5_w, conv5_b = self.conv5_hyper(context)
        conv6_w, conv6_b = self.conv6_hyper(context)
        exit_w, exit_b = self.exit_hyper(context)
        if teach:
            return conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, conv4_w, conv4_b, conv5_w, conv5_b, conv6_w, conv6_b, exit_w, exit_b
        logits = []
        if training or ensemble:
            for i in range(context.shape[0]):
                x = self.conv1_target(input, conv1_w[i], conv1_b[i])
                x = self.conv2_target(x, conv2_w[i], conv2_b[i])
                x = self.conv3_target(x, conv3_w[i], conv3_b[i])
                x = self.conv4_target(x, conv4_w[i], conv4_b[i])
                x = self.conv5_target(x, conv5_w[i], conv5_b[i])
                x = self.conv6_target(x, conv6_w[i], conv6_b[i])
                x = torch.mean(x,dim=2)
                x = self.exit_target(x,exit_w[i],exit_b[i])
                x = self.weight[i].reshape(1,-1)*x+self.bias[i].reshape(1,-1) # calibrate the logits with context
                logits.append(x)
        else:
            x = self.conv1_target(input, conv1_w[0], conv1_b[0])
            x = self.conv2_target(x, conv2_w[0], conv2_b[0])
            x = self.conv3_target(x, conv3_w[0], conv3_b[0])
            x = self.conv4_target(x, conv4_w[0], conv4_b[0])
            x = self.conv5_target(x, conv5_w[0], conv5_b[0])
            x = self.conv6_target(x, conv6_w[0], conv6_b[0])
            x = torch.mean(x,dim=2)
            x = self.exit_target(x,exit_w[0],exit_b[0])
            x = self.weight[idx].reshape(1,-1)*x+self.bias[idx].reshape(1,-1) # calibrate the logits with context
            logits = x

        if ensemble:
            ensemble_logits = torch.zeros_like(logits[0])
            for logit in logits:
                ensemble_logits += logit
            ensemble_logits /= len(logits)
            logits = ensemble_logits

        return logits