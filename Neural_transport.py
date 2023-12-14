#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2023

@author: Roberto Miele

Neural transport implementation combined with deep generative models 
(SGAN or VAE) based on "Deep generative networks for multivariate fullstack seismic data inversion using inverse autoregressive flows"
Submitted to Natural Resources Research on the 14/12/2023

========================++++++++++++++++++=====================================
MIT License

Copyright (c) 2023 Roberto Miele

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
========================++++++++++++++++++=====================================
This IAF algorithm is based on the work from 
Levy S., Laloy E., and Linde N., 2023
Variational Bayesian inference with complex geostatistical priors using inverse autoregressive flows
10.1016/j.cageo.2022.105263

              
"""

import argparse
import os
from os.path import exists
import json
import numpy as np
import time 
import copy
# libraries for neural transport
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.distributions.transforms import iterated, affine_autoregressive, block_autoregressive
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormalizingFlow, AutoIAFNormal
from functools import partial
import matplotlib.pyplot as plt
import dill as pickle
pickle.settings['recurse'] = False
from Gslib import Gslib

#import forward solvers
#import pygimli as pg
import forward_model as fm 
from FMclasses import ElasticModels

torch.pi = torch.acos(torch.zeros(1, dtype=torch.float64)).item() * 2

class Posterior(dist.TorchDistribution):
    arg_constraints = {}
    support = constraints.real_vector

    def __init__(self, setup):
        self.device = setup.device
        
        self.relative_error= setup.relative_error
        self.netG = setup.gen

        self.tt = setup.tt
        self.DGM = setup.DGM
        self.total_event_size = setup.total_event_size
        self.batch_size = setup.batch_size
        self.G = torch.Tensor(setup.A).to(self.device)
        self.d = torch.Tensor(setup.dobs.float()).to(self.device)
        self.fwd = setup.fwd
        self.prior = prior(self.total_event_size, self.batch_size)
        self.outdir= setup.outdir
        self.wavelet= setup.wavelet
        self.padding = setup.padding
        self.epoch= setup.epoch
        self.c_sigma = setup.c_sigma
        self.truemodel= setup.truemodel
        self.ipmax= setup.ipmax
        self.ipmin= setup.ipmin
        self.problem_type= args.problem_type
        self.sigma = setup.sigma
        self.noise = setup.noise
        # self.var = torch.pow(self.sigma, torch.tensor(2))
        
        super().__init__()

    @property
    def batch_shape(self):
        '''batch_size>1 doesn't work well!'''
        return (self.batch_size,)

    @property
    def event_shape(self):
        return (self.total_event_size,)

    def sample(self, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        # return torch.zeros(sample_shape + self.batch_shape + self.event_shape)
        return torch.zeros(sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, state):
        
        '''to check: should it be multiplied by the prior?'''
        # reshape state tensor generator input shape
        # znp = torch.reshape(state, [-1, self.total_event_size])
        z = state.float().to(self.device)  # send tensor to device (cpu/gpu)

        # send the states through the generator to generate samples
        model_f, model_ip = self.netG(z)
        if self.DGM == 'GAN':
            model_f = 0.5*(model_f+1)  # normalize generated model to [0,1]
        
        model_ip= (self.ipmax-self.ipmin)*model_ip+self.ipmin
        if self.problem_type!=2:
        #     pass
        # elif True:
            if self.fwd == 'fullstack':
                self.sim = fm.fullstack_solver_simple(model_ip, self)
        
        else: self.sim = model_ip
        
        
        e= self.sim.squeeze() - self.d.squeeze()
        
        se = torch.pow(e, torch.tensor(2.0))
        
        N= e.shape[-1]*e.shape[-2]
        
        first= -(N/2)*torch.log(torch.tensor(2.0)*torch.pi)
        
        second= -(1/2)*torch.sum(torch.log(self.sigma.view(-1,N)**2))
               
        num = e.view(-1,N)
        
        third= -(1/2)*torch.sum((num/self.sigma.view(-1,N))**2,dim=1)
        
        log_like = first+second+third
        
        # N = torch.tensor(e.shape[-1])

        # log_like2 = - (N / torch.tensor(2.0)) * torch.log(torch.tensor(2.0) * torch.pi)\
        #             - torch.tensor(0.5) * torch.sum(torch.log(self.sigma.view(-1,8000)**2))\
        #             - torch.tensor(0.5) * torch.sum(torch.pow(e.view(-1,8000)/self.sigma.view(-1,8000), torch.tensor(2.0)), dim=-1)
        
        rmse_d= np.sqrt(np.sum(se.detach().cpu().numpy().reshape(-1,N), axis=-1)/N).mean()
        wrmse_d= np.sqrt(torch.sum((num/self.sigma.view(-1,N))**2).detach().cpu().numpy()/N).mean()
        
        errors= np.concatenate(([rmse_d],[wrmse_d]), axis=0)[None,:]
        
        if self.epoch>0:   
             b_file= torch.load(self.outdir+'Data_e.pt')
             errors= torch.cat((b_file,torch.tensor(errors)),dim=0)
        torch.save(torch.tensor(errors), self.outdir+'Data_e.pt')

        return log_like.to(self.device) + self.prior.log_prob(state).to(self.device) 
        


def prior(total_event_size, batch_size=1):
    return dist.MultivariateNormal(torch.zeros(batch_size, total_event_size, dtype=torch.float32),
                                    torch.eye(total_event_size))

def model(setup):
    pyro.sample("Z", Posterior(setup))

def guide_Normal(setup):
    loc = pyro.param('post_loc', lambda: torch.zeros(setup.total_event_size))
    scale = pyro.param('post_scale', lambda: torch.ones(setup.total_event_size))
    pyro.sample("Z", dist.MultivariateNormal(loc, scale_tril=torch.diag(scale)))  


#%%
def main(args):
    pyro.set_rng_seed(args.rng_seed)
    workdir = args.workdir
    outdir = args.outdir
    os.chdir(workdir)
    ncase = 'noise'
    
    # NeuTra parameters
    flow = args.flow_type            # IAF, NAF or BNAF, (WARNING! BNAF cannot have arbitrary values as input to posterior.log_prob())
    batch_size = 1
    num_particles = args.num_particles
    DGM = args.DGM  # choose from: 'VAE' or 'SGAN'
    decay_lr = 1  #If to use exponential decay of the learning rate
    
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') #'cuda' 
    
    wavelet = torch.Tensor(np.genfromtxt(args.workdir+args.wavelet_file)[np.newaxis,np.newaxis,:].T).to(device)*args.Wwav
    setup = fm.setup(wavelet.float(), device)  # initialize data object containing SGD parameters
    setup.relative_error = args.relative_error 
    setup.c_sigma = args.c_sigma
    
    #%% 
    
    setup.total_event_size = 60
    if DGM == 'VAE':
        from VAE import VAE_2layers
        # load parameters of trained VAE
        dnnmodel = VAE_2layers(z_dim=60).to(device) 
        dnnmodel.load_state_dict(torch.load(args.workdir+ args.saved_state_name, map_location=device))
        for param in dnnmodel.parameters():
            param.requires_grad = False
        netG = dnnmodel.forwardNT
    
    elif DGM == 'GAN':
        from GAN import uGenerator_2layers
        dnnmodel = uGenerator_2layers(60).to(device) 
        dnnmodel.load_state_dict(torch.load(args.workdir+ args.saved_state_name, map_location=device))
        for param in dnnmodel.parameters():
            param.requires_grad = False
        netG= dnnmodel.forward
        os.chdir(workdir)
    else:
        print('not a valid DGM')

#%%  ===============================================================================================================
    setup.DGM = DGM
    setup.device = device
    setup.batch_size = batch_size
    
    Ip0= Gslib().Gslib_read(args.workdir+'/Ip_zone0.gslib').data.Ip.values
    Ip1= Gslib().Gslib_read(args.workdir+'/Ip_zone1.gslib').data.Ip.values
    setup.ipmin= Ip1.min()
    setup.ipmax= Ip0.max()
    bounds_zones={0:np.array([Ip0.min(),Ip0.max()]),
                  1:np.array([Ip1.min(),Ip1.max()])}
    ntest=1
    zs = torch.tensor(torch.load(args.workdir+'/Test_models/'+args.saved_model, map_location=device)).to(device)[np.newaxis]
    setup.zs = zs.to(device)

    # run seismic inversion based on seismic from encoded data during training       
    real_f,real_ip = netG(zs)
    real_f = torch.round(real_f)
    if DGM=='GAN':
        
        real_f = (real_f + 1) * 0.5
        real_f= real_f.detach().cpu()
    real_ip= (Ip0.max()-Ip1.min())*real_ip+Ip1.min() #denormalizing ip
    # model_ip= f'Ip_{args.saved_model[-6:]}'
    
    if args.problem_type >= 2:
        if not os.path.isdir(args.workdir+args.outdir+'/dss'):
            os.mkdir(args.workdir+args.outdir+'/dss')
        import subprocess
        args.nx=100
        args.nz=80
        Ip_models= ElasticModels(args, ipmin=Ip1.min(), ipmax= Ip0.max(),ipzones=bounds_zones) 
        
        real_f= real_f.detach().cpu()
        
        # Ip_models.writeallfac_dss(real_f.detach().cpu().numpy())
        n_sim = 1 if args.problem_type == 2 else 1
        # Ip_models.write_parfile(0,'unc', n_sim)
        # subprocess.run(args=[args.workdir+'/DSS.C.64.exe', args.workdir+'/ssdir.par'])
        
        ntest = n_sim
        
    for j in range(ntest):
        if args.problem_type==2:
            args.suboutdir = args.outdir+f'simulation_{j}/'
            if not os.path.isdir(args.workdir+args.outdir+f'simulation_{j}'):
                os.mkdir(args.workdir+args.suboutdir)
                os.mkdir(args.workdir+args.suboutdir+'/Guides')
            outdir= args.suboutdir
        
        elif args.problem_type== 3:
            real_ip= Gslib().Gslib_read(f'{args.workdir+args.outdir}/dss/ip_real_{j+1}.out').data.values
            real_ip= torch.from_numpy(np.reshape(real_ip,(real_f.shape[-2],real_f.shape[-1],))).to(device)[None,None,:]

        setup.outdir = outdir
        setup.fwd = 'fullstack'
        setup.dobs = fm.fullstack_solver_simple(real_ip, setup).to(device)
        
        setup.sigma = setup.relative_error*torch.abs(setup.dobs.detach().clone())+setup.c_sigma
        setup.noise = torch.randn_like(setup.dobs)*setup.sigma
        setup.dobs += setup.noise
        
        if args.problem_type==2: setup.dobs = real_f
        setup.truemodel = (real_f,real_ip,setup.dobs)
        
        plt.figure()
        plt.imshow(real_f.squeeze().detach().cpu(), cmap='gray')
        plt.colorbar()
        plt.savefig(outdir+'/Real_F')
        plt.close()
    
        plt.figure()
        plt.imshow(real_ip.squeeze().detach().cpu(), cmap='jet')
        plt.colorbar()
        plt.savefig(outdir+'/Real_Ip' )
        plt.close()
    
        plt.figure()
        plt.imshow(setup.dobs.squeeze().detach().cpu(), cmap='seismic')
        plt.colorbar()
        plt.savefig(outdir+'/Real_d')
        plt.close()
        
        setup.gen = netG
        setup.relative_error = torch.tensor(setup.relative_error).to(device)
        setup.flow= flow
        setup.problem_type = args.problem_type
        # Fit an autoguide
        if flow == 'NAF':
            print("\nFitting a NAF autoguide ...")
            guide = AutoNormalizingFlow(model, partial(
                iterated, args.num_flows, affine_autoregressive))
        elif flow== 'BNAF':
            print("\nFitting a BNAF autoguide ...")
            guide = AutoNormalizingFlow(model, partial(
                iterated, args.num_flows, block_autoregressive, activation='ELU'))
        elif flow == 'IAF':
            print("\nFitting a IAF autoguide ...")
            guide = AutoIAFNormal(model, hidden_dim=[setup.total_event_size*2]*args.hidden_layers, num_transforms=args.num_flows)
        elif flow == 'normal':
            print("\nFitting Standard variational inference with a Gaussian approximation")
            guide = guide_Normal
    
        if decay_lr==1:
            optimizer = torch.optim.Adam  # ({"lr": args.learning_rate})
            scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': {
                                                 'lr': args.learning_rate}, 'gamma': 0.1})
        else:
            scheduler = torch.optim.Adam({"lr": args.learning_rate})
        
        pyro.clear_param_store()
        svi = SVI(model, guide, scheduler, Trace_ELBO(
                  num_particles=num_particles, vectorize_particles=True))
        loss_vals =  []
        logS = []
        if exists(outdir+'Data_RMSE.csv'):
            os.remove(outdir+'Data_RMSE.csv') 
        start = time.time()
        for i in range(args.num_steps):
            setup.epoch=i
            loss = svi.step(setup)
            if (i+1)%50 == 0:
                # posterior = guide.get_posterior()
                # logS_val = -posterior.log_prob(torch.reshape(zs,(1,-1))).detach().numpy()[0]
                print("[{}]Elbo loss = {:.2f}".format(i+1, loss))
                # print("logS = {:.2f}".format(logS_val))
                # logS.append(np.int64(logS_val))
                if (i+1)<=350 and flow!='normal':
                    torch.save(guide.state_dict(), outdir+'saved_NT_params_step_{:}.pt'.format(i+1))
            if (i+1)%50 == 0:
                with open(outdir+'setup.pkl', 'wb') as f:
                    if setup.fwd=='nonlinear':
                        dummy = copy.copy(setup)
                        del dummy.tt
                        pickle.dump(dummy, f)
                        del dummy
                    else:
                         pickle.dump(setup, f)
                with open(outdir+'loss.pkl', 'wb') as f:
                    pickle.dump((loss_vals,logS), f)
            loss_vals.append(np.int64(loss))
        end = time.time()
        if flow!='normal':
            torch.save(guide.state_dict(), outdir+'saved_NT_params_step_last.pt')
    
        print(end - start)
    
        ''' save parameters'''
        pyro.get_param_store().save(outdir+'NeuTra_paramstore')
        with open(outdir+'setup.pkl', 'wb') as f:
            if setup.fwd=='nonlinear':
                del setup.tt
            pickle.dump(setup, f)
        with open(outdir+'loss.pkl', 'wb') as f:
            pickle.dump((loss_vals,logS), f)
        torch.save(guide, outdir+'Guides/saved_'+flow+'_guide_'+args.DGM+'.pt')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neutral transport and reparametrization"
                        )
    parser.add_argument("--restart", default=0, type=int, help="if to restrat from existing trained model"
                        )
    parser.add_argument("-n", "--num-steps", default=2001, type=int, help="number of SVI steps"
                        )
    parser.add_argument("-lr","--learning-rate",default=0.01, type=float,help="learning rate for the Adam optimizer",
                        )
    parser.add_argument("--rng-seed", default=123457, type=int, help="RNG seed"
                        )
    parser.add_argument( "--num-warmup", default=100, type=int, help="number of warmup steps for NUTS"
                        )
    parser.add_argument("--num-samples",default=500,type=int,help="number of samples to be drawn from NUTS",
                        )
    parser.add_argument("--num-flows", default=2, type=int, help="number of flows in the autoguide"
                        )
    parser.add_argument("--flow-type", default= 'IAF', type=str, help="IAF, NAF, BNAF"
                        )
    parser.add_argument("--hidden-layers", default=1, type=int, help="number of layers in each flow"
                        )
    parser.add_argument("--num-particles", default= 1, type=int, help="number of sample to train each step"
                        )
    parser.add_argument("--DGM", default= 'VAE' , type=str, help="Generative model VAE1, VAE2, GAN1 or GAN2"
                        )
    parser.add_argument("--saved-state-name", default= 'VAE_epoch1000.pth', type=str, help="Generative model VAE or GAN saved state"
                        )
    parser.add_argument("--wavelet-file", default= 'wavelet_simm.asc', type=str, help="name of the wavelet file"
                        )
    parser.add_argument("--Wwav", default= 1, type=float, help="scale amplitude content"
                        )
    parser.add_argument("--saved-model", default='mv1.pt', type=str, help="upload a model from a file, 'None' if to draw randomly"
                        )
    parser.add_argument("--problem_type", default=1,  help="1 or 2: \
                        if 1= does the method invert? inversion problem within the prior learned by the DGM;\
                        if 2 = does the method handle uncertainty? inversion problem using a different Ip simulation"
                        )
    parser.add_argument("--inversion", default='data', type=str, help="type of inversion: Data or model (pix-to-pix)"
                        )
    parser.add_argument("--c_sigma", default= 5, type=int, help="standard deviation of the noise in [amp]"
                        )
    parser.add_argument("--relative_error", default=0.03, type=int, help="relative error of amplitudes [n%]"
                        )
    parser.add_argument("--workdir", default='C:/..../', 
                        type=str, help="working directory (where all the models are"
                        )
    parser.add_argument("--outdir", default= 'output', type=str, help="directory to save results in"
                        )
    #fill below only for problem type 2
    parser.add_argument("--null_val", default=-9999.00, 
                        type=float, help='null value in well data'
                        )
    parser.add_argument("--var_N_str", default=[1,1], 
                        type=int, help='number of variogram structures per facies [fac0, fac1,...]'
                        )
    parser.add_argument("--var_nugget", default=[0,0],
                        type=float, help='variogram nugget per facies [fac0, fac1,...]'
                        )
    parser.add_argument("--var_type", default=[[1],[1]], 
                        type=int, help='variogram type per facies [fac0[str1,str2,...], fac1[str1,str2,...],...]: 1=spherical,2=exponential,3=gaussian'
                        )
    parser.add_argument("--var_ang", default=[[0,0],[0,0]],
                        type=float, help='variogram angles per facies [fac0[angX,angZ], fac1[angX,angZ],...]'
                        )
    parser.add_argument("--var_range", default=[[[80,30]],[[80,50]]],
                        type=float, help='variogram ranges per structure and per facies [fac0[str0[rangeX,rangeZ],str1[rangeX,rangeZ]...],fac1[str1[rangeX,rangeZ],...],...]'
                        )
    
    import sys
    sys.path.append(parser.parse_args().workdir)
    
    out_folder= parser.parse_args().outdir+'_2'
    
    # Generating output folder
    if not os.path.isdir(parser.parse_args().workdir+out_folder):
        os.mkdir(parser.parse_args().workdir+out_folder)
        os.mkdir(parser.parse_args().workdir+out_folder+'/Guides')
    
    args = parser.parse_args()
    args.outdir= out_folder+'/'
    with open(args.workdir+args.outdir+'/run_commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    main(args)
