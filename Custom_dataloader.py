# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:07:19 2023
    Custom dataloader for Norne
@author: Roberto.Miele
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob
import torchvision
from Gslib import Gslib


class FaciesSeismicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, nsim, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.lendata = os.listdir(root_dir+'/Facies_TI')
        self.transform = transform
        self.nsim= nsim
        

    def __len__(self):
        return len(self.lendata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fac_filename= self.root_dir+'Facies_TI/'+f'{idx}.pt'
        seis_filename= self.root_dir+'Ip_TI/'+f'{idx}_{np.random.randint(0,self.nsim)}.pt'
        fac_file= torch.load(fac_filename)
        seis_file= torch.load(seis_filename)
        
        fac_file[fac_file==0]=-1
        sample = (fac_file, seis_file)

        return sample
    
    
def precompute_TI(args, transforms, Ip_models):
    print('Pre-computing TI Seismic')
    import subprocess
    import os 
    if args.precomputed==True: 
        print('Already pre-computed')
        return FaciesSeismicDataset(args.project_path+args.TI_path,args.nsim)
    else:
        if os.path.isdir(args.project_path+args.TI_path+'/Ip_TI'): 
            shutil.rmtree(args.project_path+args.TI_path+'/Ip_TI')
        if os.path.isdir(args.project_path+args.TI_path+'/Facies_TI'): 
            shutil.rmtree(args.project_path+args.TI_path+'/Facies_TI')
    
        dataset = torchvision.datasets.ImageFolder(root=args.project_path+args.TI_path, transform=transforms)
        N= len(dataset)
        dataloader= torch.utils.data.DataLoader(dataset=dataset, batch_size=N)
        
        os.mkdir(args.project_path+args.TI_path+'/Ip_TI')
        os.mkdir(args.project_path+args.TI_path+'/Facies_TI') 
            
        for _,data in enumerate(dataloader):
            data= (data[0][:,0,None,:,:]+1)*0.5
            Ip_models.writeallfac_dss(data.detach().cpu().numpy())  #write TI facies in gslib format
            
            Ip_models.simulations= torch.zeros((args.nsim, 1, data.shape[2], data.shape[3])) 
            for i in range(N):
                Ip_models.write_parfile(i,'unc',args.nsim) #write parfile
                subprocess.run(args=[f'{Ip_models.inf}DSS.C.64.exe', f'{Ip_models.inf}ssdir.par'], stdout=subprocess.DEVNULL) #run DSS
                
                for ssi in range (args.nsim):
                    #read simulation
                    Ip_models.simulations[ssi]= torch.from_numpy(np.reshape(Gslib().Gslib_read(f'{Ip_models.ouf}/dss/ip_real_{ssi+1}.out').data.values.squeeze(),
                                  (1, args.nz, args.nx)))
                
                    torch.save(Ip_models.simulations[ssi], args.project_path+args.TI_path+f'/Ip_TI/{i}_{ssi}.pt')
                dtw= torch.from_numpy(data[i].detach().cpu().numpy())
                torch.save(dtw, args.project_path+args.TI_path+f'/Facies_TI/{i}.pt')
        
        dataset= FaciesSeismicDataset(args.project_path+args.TI_path,args.nsim)
        return dataset
