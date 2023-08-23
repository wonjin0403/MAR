import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt
from skimage.transform import resize

import numpy as np

import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd import Function


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def conv_block(in_dim,out_dim,act_fn,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=stride, padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
    )
    return model
    
class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3
    
def conv_2d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_dim),
        act_fn
    )
    return model


class UNet(nn.Module):

    def __init__(self, input_dim, num_feature, output_dim, out_clamp):
        super(UNet, self).__init__()

        self.in_dim = input_dim
        self.out_dim = num_feature
        self.final_out_dim = output_dim
        self.out_clamp = out_clamp

        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ELU(inplace=True)
        act_fn_R = nn.ReLU(inplace=True)

        # encoder
        self.down_1_1 = conv_2d(self.in_dim, self.out_dim, act_fn_2)
        self.down_1_2 = conv_2d(self.out_dim, self.out_dim, act_fn_2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        
        self.down_2_1 = conv_2d(self.out_dim, self.out_dim * 2, act_fn_2)
        self.down_2_2 = conv_2d(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        
        self.down_3_1 = conv_2d(self.out_dim * 2, self.out_dim * 4, act_fn_2)
        self.down_3_2 = conv_2d(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        
        self.down_4_1 = conv_2d(self.out_dim * 4, self.out_dim * 8, act_fn_2)
        self.down_4_2 = conv_2d(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.pool_4 = nn.MaxPool2d(kernel_size=2)

        self.down_5_1 = conv_2d(self.out_dim * 8, self.out_dim * 16, act_fn_2)

        # decoder
        self.up_5_1 = conv_2d(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        
        self.unpool_4 = nn.ConvTranspose2d(self.out_dim * 8, self.out_dim * 8, kernel_size=2, stride=2, padding=0, bias=True)
        
        self.up_4_2 = conv_2d(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_4_1 = conv_2d(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        
        self.unpool_3 = nn.ConvTranspose2d(self.out_dim * 4, self.out_dim * 4, kernel_size=2, stride=2, padding=0, bias=True)
        
        self.up_3_2 = conv_2d(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_3_1 = conv_2d(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        
        self.unpool_2 = nn.ConvTranspose2d(self.out_dim * 2, self.out_dim * 2, kernel_size=2, stride=2, padding=0, bias=True)
        
        self.up_2_2 = conv_2d(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_2_1 = conv_2d(self.out_dim * 2, self.out_dim * 1, act_fn_2)
        
        self.unpool_1 = nn.ConvTranspose2d(self.out_dim * 1, self.out_dim * 1, kernel_size=2, stride=2, padding=0, bias=True)
        
        self.up_1_2 = conv_2d(self.out_dim * 2, self.out_dim * 1, act_fn_2)
        self.up_1_1 = conv_2d(self.out_dim * 1, self.out_dim * 1, act_fn_2)

        # output
        self.out1 = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.out2 = nn.Sigmoid()
        self.out3 = nn.Tanh()

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)
                
        print("------U-Net Init Done------")

    def forward(self, input):      
        down_1_1 = self.down_1_1(input)
        down_1_2 = self.down_1_2(down_1_1)
        pool_1 = self.pool_1(down_1_2)
        
        down_2_1 = self.down_2_1(pool_1)
        down_2_2 = self.down_2_2(down_2_1)
        pool_2 = self.pool_2(down_2_2)
        
        down_3_1 = self.down_3_1(pool_2)
        down_3_2 = self.down_3_2(down_3_1)
        pool_3 = self.pool_3(down_3_2)
        
        down_4_1 = self.down_4_1(pool_3)
        down_4_2 = self.down_4_2(down_4_1)
        pool_4 = self.pool_4(down_4_2)

        down_5_1 = self.down_5_1(pool_4)
#        return down_5_1, down_4_2, down_3_2, down_2_2, down_1_2
        # enc_dec_outline
#    def forward1(self, down_5_1, down_4_2, down_3_2, down_2_2, down_1_2)
        up_5_1 = self.up_5_1(down_5_1)

        unpool_4 = self.unpool_4(up_5_1)
        cat4 = torch.cat((unpool_4, down_4_2), dim=1)
        up_4_2 = self.up_4_2(cat4)
        up_4_1 = self.up_4_1(up_4_2)
        
        unpool_3 = self.unpool_3(up_4_1)
        cat3 = torch.cat((unpool_3, down_3_2), dim=1)
        up_3_2 = self.up_3_2(cat3)
        up_3_1 = self.up_3_1(up_3_2)
        
        unpool_2 = self.unpool_2(up_3_1)
        cat2 = torch.cat((unpool_2, down_2_2), dim=1)
        up_2_2 = self.up_2_2(cat2)
        up_2_1 = self.up_2_1(up_2_2)
        
        unpool_1 = self.unpool_1(up_2_1)
        cat1 = torch.cat((unpool_1, down_1_2), dim=1)
        up_1_2 = self.up_1_2(cat1)
        up_1_1 = self.up_1_1(up_1_2)       
        
        out = self.out1(up_1_1)
        # out = self.out2(out)
        out = self.out3(out)

        # if self.out_clamp is not None:
        #     out = torch.clamp(out, min=self.out_clamp[0], max=self.out_clamp[1])

        return out
    

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    input_ = torch.randn(9, 1, 128, 128).cuda()

    Arg = namedtuple('Arg', ['input_dim', 'num_feature', 'output_dim', 'out_clamp'])
    args = Arg(1, 16, 1, None)
    m  = UNet(args).cuda()
    output = m(input_)

    print("output shape : ", output.shape)
