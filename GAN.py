# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:59:42 2022
    This file contains the Generators and the Discriminators for all the cases
    both conditional and unconditional. 
    
    UNet Generator and Unconditional Generators and Discriminators are from Zhang et al., 2021
@author: roberto.miele
"""
import torch
from torch import nn

class YNetDiscriminator(nn.Module):
    def __init__(self, nw=1):
        super(YNetDiscriminator, self).__init__()
        
        def block(in_channel, out_channel, ks, normalize=True,relu=True, sig=False):
            layers = []

            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=2, bias=False))
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            if relu:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sig:
                layers.append(nn.Sigmoid())

            return layers
        
        self.well= nw
        
        self.DnetS0 = nn.Sequential(*block(1, 64, 4, normalize=False))
        self.DnetF0 = nn.Sequential(*block(nw, 64, 4, normalize=False))
        self.DnetJ0 = nn.Sequential(*block(2, 64, 4, normalize=False))

        self.DnetS1 = nn.Sequential(*block(64, 128, 3))
        self.DnetF1 = nn.Sequential(*block(64, 128,3))
        self.DnetJ1 = nn.Sequential(*block(64*3, 128,3))

        self.DnetS2 = nn.Sequential(*block(128, 256,(3,4)))
        self.DnetF2 = nn.Sequential(*block(128, 256,(3,4)))
        self.DnetJ2 = nn.Sequential(*block(128*3, 256,(3,4)))
        
        self.DnetF3 = nn.Sequential(*block(256, 1, (3,5), normalize=False, relu=False, sig=True))
        self.DnetJ3 = nn.Sequential(*block(256*3, 1,(3,5), normalize=False, relu=False, sig=True))

    def forward(self, facies, seismic, well=None):
        
        j0= torch.cat([facies, seismic], 1)
        
        if well!=None: f0= torch.cat([well, facies], 1)
        else: f0= facies
            
        f1 = self.DnetF0(f0)
        s1 = self.DnetS0(seismic)
        j1 = self.DnetJ0(j0)
        
        j1= torch.cat([f1,s1,j1], 1)
        f2 = self.DnetF1(f1)
        s2 = self.DnetS1(s1)
        j2 = self.DnetJ1(j1)
        
        j2= torch.cat([f2,s2,j2], 1)
        f3 = self.DnetF2(f2)
        s3 = self.DnetS2(s2)
        j3 = self.DnetJ2(j2)
        
        j3= torch.cat([f3,s3,j3], 1)
        scoreJ= self.DnetJ3(j3)
        scoreF= self.DnetF3(f3)
        
        return scoreF.reshape(-1, 16).mean(dim=1).reshape(-1, 1), scoreJ.reshape(-1, 16).mean(dim=1).reshape(-1, 1)

class cDiscriminator(nn.Module):
    def __init__(self):
        super(cDiscriminator, self).__init__()

        def block(in_channel, out_channel, ks, normalize=True,relu=True, sig=False):
            layers = []

            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=2, bias=False))
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            if relu:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sig:
                layers.append(nn.Sigmoid())

            return layers
        self.Dnet0 = nn.Sequential(*block(2, 64, 4, normalize=False))
        self.Dnet1 = nn.Sequential(*block(64, 128, 3))
        self.Dnet2 = nn.Sequential(*block(128, 256, (3, 4)))
        self.Dnet3 = nn.Sequential(*block(256, 1, (3, 5), normalize=False, relu=False, sig=True))

    def forward(self, image, mask):
        x = torch.cat([image, mask], 1)
        x1 = self.Dnet0(x)
        x2 = self.Dnet1(x1)
        x3 = self.Dnet2(x2)
        score = self.Dnet3(x3)
                
        score = score.reshape(-1, 16).mean(dim=1).reshape(-1, 1)
        
        del x, x1, x2, x3
        return score

class uGenerator(nn.Module):
    def __init__(self, nz):
        super(uGenerator, self).__init__()

        self.Gnet = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, kernel_size=4,stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 256, kernel_size=5,stride=(2, 3), padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 512, kernel_size=5,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 128, kernel_size=4,stride=2, padding=(0, 1), bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 2, kernel_size=4,stride=2, padding=1, bias=False),
            )

    def forward(self, z):
        z = z.reshape(-1, z.shape[-1], 1, 1)
        out= self.Gnet(z)
        
        facies= nn.Tanh()(out[:,0,None,:])
        ip= nn.Sigmoid()(out[:,1,None,:])
        return facies, ip


class uGenerator_2layers(nn.Module):
    def __init__(self,zz):
        super(uGenerator_2layers, self).__init__()

        self.Gnet_main = nn.Sequential(
            nn.ConvTranspose2d(zz, 512, kernel_size=4,stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 256, kernel_size=5,stride=(2, 3), padding=1, bias=False),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 512, kernel_size=5,stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 128, kernel_size=4,stride=2, padding=(0, 1), bias=False),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(64),nn.LeakyReLU(0.2))

        self.fnet = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=2,stride=1,padding=0,bias=False),
                                  nn.LeakyReLU(0.2),
                                  nn.ConvTranspose2d(32,1,kernel_size=2,stride=1,padding=1,bias=False),
                                  nn.Tanh())
        
        self.ipnet = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=2,stride=1,padding=0,bias=False),
                                  nn.LeakyReLU(0.2),
                                  nn.ConvTranspose2d(32,1,kernel_size=2,stride=1,padding=1,bias=False),
                                  nn.Sigmoid())
    def forward(self, z):
        z = z.reshape(-1, z.shape[-1], 1, 1)
        out= self.Gnet_main(z)
        facies = self.fnet(out)
        ip = self.ipnet(out)
        
        return facies, ip




class uDiscriminator_1I(nn.Module):
    def __init__(self):
        super(uDiscriminator_1I, self).__init__()

        def block(in_channel, out_channel, ks, normalize=True,relu=True, sig=False):
            layers = []

            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=2, bias=False))
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            if relu:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sig:
                layers.append(nn.Sigmoid())

            return layers

        self.Dnet0 = nn.Sequential(*block(1, 64, 4, normalize=False))
        self.Dnet1 = nn.Sequential(*block(64, 128,3))
        self.Dnet2 = nn.Sequential(*block(128, 256, (3,4)))
        self.Dnet3 = nn.Sequential(*block(256, 1, (3,5), normalize=False, relu=False, sig=True))


    def forward(self, image):
    
        x1 = self.Dnet0(image)
        x2 = self.Dnet1(x1)
        x3 = self.Dnet2(x2)
        score = self.Dnet3(x3)
        
        del x1, x2, x3
        return score.reshape(-1, 16).mean(dim=1).reshape(-1, 1)

class uDiscriminator(nn.Module):
    def __init__(self, nw):
        super(uDiscriminator, self).__init__()

        self.Dnet = nn.Sequential(
            nn.Conv2d(nw, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.Dnet(x).reshape(-1,1)



class WNetDiscriminator(nn.Module):
    def __init__(self, nw=1):
        super(WNetDiscriminator, self).__init__()
        
        def block(in_channel, out_channel, ks, normalize=True,relu=True, sig=False):
            layers = []

            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=2, bias=False))
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channel))
            if relu:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sig:
                layers.append(nn.Sigmoid())

            return layers
        
        self.DnetS0 = nn.Sequential(*block(1, 64, 4, normalize=False))
        self.DnetF0 = nn.Sequential(*block(nw, 64, 4, normalize=False))
        self.DnetJ0 = nn.Sequential(*block(2, 64, 4, normalize=False))

        self.DnetS1 = nn.Sequential(*block(64, 128, 3))
        self.DnetF1 = nn.Sequential(*block(64, 128,3))
        self.DnetJ1 = nn.Sequential(*block(64*3, 128,3))

        self.DnetS2 = nn.Sequential(*block(128, 256,(3,4)))
        self.DnetF2 = nn.Sequential(*block(128, 256,(3,4)))
        self.DnetJ2 = nn.Sequential(*block(128*3, 256,(3,4)))
        
        self.DnetF3 = nn.Sequential(*block(256, 1, (3,5), normalize=False, relu=False, sig=True))
        self.DnetS3 = nn.Sequential(*block(256, 1, (3,5), normalize=False, relu=False, sig=True))
        self.DnetJ3 = nn.Sequential(*block(256*3, 1,(3,5), normalize=False, relu=False, sig=True))

    def forward(self, facies, seismic, well=None):
        
        j0= torch.cat([facies, seismic], 1)
           
        f1 = self.DnetF0(facies)
        s1 = self.DnetS0(seismic)
        j1 = self.DnetJ0(j0)
        del facies, seismic, j0
        j1= torch.cat([f1,s1,j1], 1)
        f2 = self.DnetF1(f1)
        s2 = self.DnetS1(s1)
        j2 = self.DnetJ1(j1)
        del f1,s1,j1
        j2= torch.cat([f2,s2,j2], 1)
        f3 = self.DnetF2(f2)
        s3 = self.DnetS2(s2)
        j3 = self.DnetJ2(j2)
        
        j3= torch.cat([f3,s3,j3], 1)
        scoreJ= self.DnetJ3(j3)
        scoreF= self.DnetF3(f3)
        scoreS= self.DnetF3(s3)
        
        return scoreF.reshape(-1, 16).mean(dim=1).reshape(-1, 1), scoreJ.reshape(-1, 16).mean(dim=1).reshape(-1, 1), scoreS.reshape(-1, 16).mean(dim=1).reshape(-1, 1)


class YNetDiscriminator_1(nn.Module):
    def __init__(self, nw=1):
        super(YNetDiscriminator_1, self).__init__()
        
        def block(in_channel, out_channel, ks, normalize=True,relu=True, sig=False):
            layers = []

            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=2, bias=False))
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            if relu:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sig:
                layers.append(nn.Sigmoid())

            return layers
        
        self.well= nw
        
        self.DnetS0 = nn.Sequential(*block(1, 64, 4, normalize=False))
        self.DnetF0 = nn.Sequential(*block(nw, 64, 4, normalize=False))
        self.DnetJ0 = nn.Sequential(*block(2, 64, 4, normalize=False))

        self.DnetS1 = nn.Sequential(*block(64, 128, 3))
        self.DnetF1 = nn.Sequential(*block(64, 128,3))
        self.DnetJ1 = nn.Sequential(*block(64*3, 128,3))

        self.DnetS2 = nn.Sequential(*block(128, 256,(3,4)))
        self.DnetF2 = nn.Sequential(*block(128, 256,(3,4)))
        self.DnetJ2 = nn.Sequential(*block(128*3, 256,(3,4)))
        
        self.DnetJ3 = nn.Sequential(*block(256*3, 1,(3,5), normalize=False, relu=False, sig=True))

    def forward(self, facies, seismic, well=None):
        
        j0= torch.cat([facies, seismic], 1)
        
        if well!=None: f0= torch.cat([well, facies], 1)
        else: f0= facies
            
        f1 = self.DnetF0(f0)
        s1 = self.DnetS0(seismic)
        j1 = self.DnetJ0(j0)
        
        j1= torch.cat([f1,s1,j1], 1)
        f2 = self.DnetF1(f1)
        s2 = self.DnetS1(s1)
        j2 = self.DnetJ1(j1)
        
        j2= torch.cat([f2,s2,j2], 1)
        f3 = self.DnetF2(f2)
        s3 = self.DnetS2(s2)
        j3 = self.DnetJ2(j2)
        
        j3= torch.cat([f3,s3,j3], 1)
        scoreJ= self.DnetJ3(j3)
        
        return scoreJ.reshape(-1, 16).mean(dim=1).reshape(-1, 1)