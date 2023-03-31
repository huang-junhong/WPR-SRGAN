import torch.nn as nn
import torch
from torch.nn import functional as F
import pywt
import numpy as np
from torch.nn.utils import spectral_norm

class ResUnit(nn.Module):
    def __init__(self, ksize=3, wkdim=64):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize/2))
        self.active = nn.PReLU()
        self.conv2 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize/2))

    def forward(self, input):
        current = self.conv1(input)
        current = self.active(current)
        current = self.conv2(current)
        current = input + current
        return current


class UPN(nn.Module):
    def __init__(self, indim=64, scale=2):
        super(UPN, self).__init__()
        self.conv = nn.Conv2d(indim, indim*(scale**2), 3, 1, 1)
        self.Upsample = nn.PixelShuffle(scale)
        self.active = nn.PReLU()

    def forward(self, input):
        current = self.conv(input)
        current = self.Upsample(current)
        current = self.active(current)
        return current

class SRRes(nn.Module):
    def __init__(self, wkdim=64, num_block=16):
        super(SRRes, self).__init__()
        self.head = nn.Conv2d(3, wkdim, 9, 1, 4)
        self.resblock = self._make_resblocks(wkdim, num_block)
        self.gate = nn.Conv2d(wkdim, wkdim, 3, 1, 1)
        self.up_1 = UPN(wkdim)
        self.up_2 = UPN(wkdim)

        self.comp = nn.Conv2d(wkdim*2, wkdim, 3, 1, 1)

        self.tail = nn.Conv2d(wkdim, 3, 9, 1, 4)

    def _make_resblocks(self, wkdim, num_block):
        layers = []
        for i in range(1, num_block+1):
            layers.append(ResUnit(wkdim=wkdim))
        return nn.Sequential(*layers)

    def forward(self, input):
        F_0 = self.head(input)
        current = self.resblock(F_0)
        current = self.gate(current)
        current = F_0 + current
        UP_1 = self.up_1(current)
        UP_2 = self.up_2(UP_1)

        current = self.tail(UP_2)
        return current

class SRRes2(nn.Module):
    def __init__(self, wkdim=64, num_block=16):
        super(SRRes2, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(3, wkdim, 4, 1, 1),
                                  nn.PReLU(),
                                  nn.Conv2d(wkdim, wkdim, 4, 1, 1),
                                  nn.PReLU(),
                                  nn.Conv2d(wkdim, wkdim, 3, 1, 1))
        self.resblock = self._make_resblocks(wkdim, num_block)
        self.gate = nn.Conv2d(wkdim, wkdim, 3, 1, 1)
        self.up_1 = UPN(wkdim)
        self.up_2 = UPN(wkdim)

        self.comp = nn.Conv2d(wkdim*2, wkdim, 3, 1, 1)

        self.tail = nn.Sequential(nn.Conv2d(wkdim, wkdim, 3, 1, 1),
                                  nn.PReLU(),
                                  nn.Conv2d(wkdim, 3, 3, 1, 1))

    def _make_resblocks(self, wkdim, num_block):
        layers = []
        for i in range(1, num_block+1):
            layers.append(ResUnit(wkdim=wkdim))
        return nn.Sequential(*layers)

    def forward(self, input):
        F_0 = self.head(input)
        current = self.resblock(F_0)
        current = self.gate(current)
        current = F_0 + current
        UP_1 = self.up_1(current)
        UP_2 = self.up_2(UP_1)

        current = self.tail(UP_2)
        return current

class Stander_Discriminator(nn.Module):
    def __init__(self):
        super(Stander_Discriminator,self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(64, 64, 4, 2, 1),
                                     nn.InstanceNorm2d(64),
                                     nn.LeakyReLU())
        self.layer_2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                     nn.InstanceNorm2d(128),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(128, 128, 4, 2, 1),
                                     nn.InstanceNorm2d(128),
                                     nn.LeakyReLU())
        self.layer_3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1),
                                     nn.InstanceNorm2d(256),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(256, 256, 4, 2, 1),
                                     nn.InstanceNorm2d(256),
                                     nn.LeakyReLU())
        self.layer_4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(512, 512, 4, 2, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.LeakyReLU())
        self.layer_5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(512, 512, 4, 2, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.LeakyReLU())
        self.tail    = nn.Sequential(nn.Conv2d(512,512,4,2,1),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(512,512,3,1,1),
                                     nn.LeakyReLU(),
                                     nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Conv2d(512,1,1,1,0))
    def forward(self, input):
        current = self.layer_1(input)
        current = self.layer_2(current)
        current = self.layer_3(current)
        current = self.layer_4(current)
        current = self.layer_5(current)
        current = self.tail(current)
        current = current.squeeze(1)
        return current

class Stander_Discriminator_SN(nn.Module):
    def __init__(self):
        super(Stander_Discriminator,self).__init__()
        self.norm = nn.utils.spectral_norm

        self.conv_0 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv_1 = self.norm(nn.Conv2d(64, 64, 4, 2, 1))

        self.conv_2 = self.norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv_3 = self.norm(nn.Conv2d(128, 128, 4, 2, 1))

        self.conv_4 = self.norm(nn.Conv2d(128, 128, 3, 1, 1))
        self.conv_5 = self.norm(nn.Conv2d(128, 256, 4, 2, 1))

        self.conv_6 = self.norm(nn.Conv2d(256, 256, 3, 1, 1))
        self.conv_7 = self.norm(nn.Conv2d(256, 512, 4, 2, 1))

        self.conv_8 = self.norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv_9 = self.norm(nn.Conv2d(512, 512, 4, 2, 1))

        self.tail   = nn.Sequential(nn.Conv2d(512,512,4,2,1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(512,512,3,1,1),
                                    nn.LeakyReLU(),
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Conv2d(512,1,1,1,0))

    def forward(self, input):

        x = F.leaky_relu(self.conv_0(input), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv_3(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv_4(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv_5(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv_6(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv_7(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv_8(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv_9(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.tail(x), negative_slope=0.2, inplace=True)
        x = x.squeeze(1)

        return x

        



class Unet_Discriminator(nn.Module):
    def __init__(self, indim=3, wkdim=64):
        super(Unet_Discriminator, self).__init__()


        self.layer_1 = nn.Sequential(nn.Conv2d(indim, wkdim, 3, 1, 1),
                                     nn.InstanceNorm2d(wkdim),
                                     nn.LeakyReLU())

        self.layer_2 = nn.Sequential(nn.Conv2d(wkdim, wkdim, 4, 2, 1),
                                     nn.InstanceNorm2d(wkdim),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(wkdim, wkdim*2, 3, 1, 1),
                                     nn.InstanceNorm2d(wkdim*2),
                                     nn.LeakyReLU())

        self.layer_3 = nn.Sequential(nn.Conv2d(wkdim*2, wkdim*2, 4, 2, 1),
                                     nn.InstanceNorm2d(wkdim*2),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(wkdim*2, wkdim*4, 3, 1, 1),
                                     nn.InstanceNorm2d(wkdim*4),
                                     nn.LeakyReLU())

        self.layer_4 = nn.Sequential(nn.Conv2d(wkdim*4, wkdim*4, 4, 2, 1),
                                     nn.InstanceNorm2d(wkdim*4),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(wkdim*4, wkdim*8, 3, 1, 1),
                                     nn.InstanceNorm2d(wkdim*8),
                                     nn.LeakyReLU())

        self.layer_global = nn.Sequential(nn.Conv2d(wkdim*8, wkdim*8, 3, 1, 1),
                                          nn.LeakyReLU(),
                                          nn.AdaptiveAvgPool2d((1,1)),
                                          nn.Conv2d(wkdim*8, wkdim, 1, 1, 0),
                                          nn.LeakyReLU(),
                                          nn.Conv2d(wkdim, 1, 1, 1, 0))

        self.layer_4v = nn.Sequential(nn.Conv2d(wkdim*8, wkdim*8, 3, 1, 1),
                                      nn.InstanceNorm2d(wkdim*8),
                                      nn.LeakyReLU(),
                                      nn.ConvTranspose2d(wkdim*8, wkdim*4, 4, 2, 1),
                                      nn.InstanceNorm2d(wkdim*4),
                                      nn.LeakyReLU())

        self.layer_3v = nn.Sequential(nn.Conv2d(wkdim*8, wkdim*4, 3, 1, 1),
                                      nn.InstanceNorm2d(wkdim*4),
                                      nn.LeakyReLU(),
                                      nn.ConvTranspose2d(wkdim*4, wkdim*2, 4, 2, 1),
                                      nn.InstanceNorm2d(wkdim*2),
                                      nn.LeakyReLU())
        self.layer_2v = nn.Sequential(nn.Conv2d(wkdim*4, wkdim*2, 3, 1, 1),
                                      nn.InstanceNorm2d(wkdim*2),
                                      nn.LeakyReLU(),
                                      nn.ConvTranspose2d(wkdim*2, wkdim, 4, 2, 1),
                                      nn.InstanceNorm2d(wkdim),
                                      nn.LeakyReLU())

        self.layer_1v = nn.Sequential(nn.Conv2d(wkdim*2, wkdim, 3, 1, 1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(wkdim, 1, 3, 1, 1))

    def forward(self, input):
        layer_1 = self.layer_1(input)
        layer_2 = self.layer_2(layer_1)
        layer_3 = self.layer_3(layer_2)
        layer_4 = self.layer_4(layer_3)
        layer_4v = self.layer_4v(layer_4)
        layer_3v = self.layer_3v(torch.cat([layer_4v, layer_3], 1))
        layer_2v = self.layer_2v(torch.cat([layer_3v, layer_2], 1))
        layer_1v = self.layer_1v(torch.cat([layer_2v, layer_1], 1))

        return layer_1v


if __name__ == '__main__':
    x = torch.randn(2,3,128,128)

