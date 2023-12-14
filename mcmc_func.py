# -*- coding: utf-8 -*-
"""

MCMC functions updated for Neural transport implementation combined with deep generative models 
(SGAN or VAE) based on "Deep generative networks for multivariate fullstack seismic data inversion using inverse autoregressive flows"
Submitted to Natural Resources Research on the 14/12/2023
modifications Roberto Miele <roberto.miele@tecnico.ulisboa.pt>

original author: Eric Laloy <elaloy@sckcen.be>, January 2017 / February 2018.
===
Copyright (C) 2023  Roberto Miele

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ===                               

"""
import numpy as np
from scipy.stats import multivariate_normal
import time
from joblib import Parallel, delayed
import sys
from functools import reduce
from scipy.stats import triang

import torch
from scipy.signal import medfilt

#import pygimli as pg
#from run_pygimli import pygimli_fw

# !!!! adapted for fullstack
from forward_model import reflectivity_ip
from torch.nn.functional import conv2d

def lhs(minn,maxn,N): # Latin Hypercube sampling
    # Here minn and maxn are assumed to be 1xd arrays 
    x = np.zeros((N,minn.shape[1]))

    for j in range (0,minn.shape[1]):
    
        idx = np.random.permutation(N)+0.5
        P =(idx - x[:,j])/N
        x[:,j] = minn[0,j] + P*(maxn[0,j] - minn[0,j])

    return x
    
def GenCR(MCMCPar,pCR):

    if type(pCR) is np.ndarray:
        p=np.ndarray.tolist(pCR)[0]
    else:
        p=pCR
    CR=np.zeros((MCMCPar.seq * MCMCPar.steps),dtype=np.float)
    L =  np.random.multinomial(MCMCPar.seq * MCMCPar.steps, p, size=1)
    L2 = np.concatenate((np.zeros((1),dtype=np.int), np.cumsum(L)),axis=0)

    r = np.random.permutation(MCMCPar.seq * MCMCPar.steps)

    for zz in range(0,MCMCPar.nCR):
        
        i_start = L2[zz]
        i_end = L2[zz+1]
        idx = r[i_start:i_end]
        CR[idx] = np.float(zz+1)/MCMCPar.nCR
        
    CR = np.reshape(CR,(MCMCPar.seq,MCMCPar.steps))
    return CR, L

def CalcDelta(nCR,delta_tot,delta_normX,CR):
    # Calculate total normalized Euclidean distance for each crossover value
    
    # Derive sum_p2 for each different CR value 
    for zz in range(0,nCR):
    
        # Find which chains are updated with zz/MCMCPar.nCR
        idx = np.argwhere(CR==(1.0+zz)/nCR);idx=idx[:,0]
    
        # Add the normalized squared distance tot the current delta_tot;
        delta_tot[0,zz] = delta_tot[0,zz] + np.sum(delta_normX[idx])
    
    return delta_tot

def AdaptpCR(seq,delta_tot,lCR,pCR_old):
    
    if np.sum(delta_tot) > 0:
        pCR = seq * (delta_tot/lCR) / np.sum(delta_tot)
        pCR = pCR/np.sum(pCR)
        
    else:
        pCR=pCR_old
    
    return pCR    

def CompLikelihood(X,fx,MCMCPar,Measurement,Extra):
    
    if MCMCPar.lik==0: # fx contains log-density
        of = np.exp(fx)       
        log_p= fx

    elif MCMCPar.lik==1: # fx contains density
        of = fx       
        log_p= np.log(of)
        
    elif MCMCPar.lik < 4: # fx contains  simulated data
        if MCMCPar.lik_sigma_est==True: # Estimate sigma model
            Sigma_res=10**(X[:,-1]) # Sigma_model is last element of X
            Sigma_meas=Measurement.Sigma*np.ones((MCMCPar.seq))
            Sigma=Sigma_res#+Sigma_meas

        else:
            if MCMCPar.lik!= 3.5: Sigma=Measurement.Sigma*np.ones((MCMCPar.seq))
        
        of=np.zeros((fx.shape[0],1))
        
        if MCMCPar.lik==3.5:  
            rof=np.zeros((fx.shape[0],1))

        log_p=np.zeros((fx.shape[0],1))
        for ii in range(0,fx.shape[0]):
            
            e = fx[ii,:].squeeze() - Measurement.MeasData.squeeze().cpu().numpy()
            
            if MCMCPar.lik==3.5: 
                re = np.abs(e/Measurement.MeasData.squeeze()).reshape(e.shape[-1]*e.shape[-2])
                # re = re[np.abs(Measurement.MeasData.squeeze().reshape(8000))>1]
                rof[ii,0]= re.mean() # e is a vector and not a 1 x d array 
                del re
                
            of[ii,0]=np.sqrt(np.sum(np.power(e,2.0))/len(e.flatten())) # e is a vector and not a 1 x d array 

            if MCMCPar.lik==2: # Compute standard uncorrelated and homoscedastic Gaussian log-likelihood
                log_p[ii,0]= - ( Measurement.N / 2.0) * np.log(2.0 * np.pi) - Measurement.N * np.log( Sigma[ii] ) - 0.5 * np.power(Sigma[ii],-2.0) * np.sum( np.power(e,2.0))
            if MCMCPar.lik==3: # Box and Tiao (1973) log-likelihood formulation with Sigma integrated out based on prior of the form p(sigma) ~ 1/sigma
                log_p[ii,0]= - ( Measurement.N / 2.0) * np.log(np.sum(np.power(e,2.0))) 
            
            if MCMCPar.lik== 3.5: # log-likelihood is calculated on relative error (standard uncorrelated and heteroscedastic Gaussian log-likelihood)
                
                log_p[ii,0]= - (Measurement.N / 2.0) * np.log(2.0 * np.pi)\
                             - 0.5 * np.sum(np.log(Measurement.sigma.squeeze().numpy()**2)) \
                             - 0.5 * np.sum(np.power((e/Measurement.sigma.squeeze().numpy()),2.0))    
                
                # # Sigma= torch.tensor(np.abs(Measurement.MeasData))*Measurement.relative_error + Measurement.c_sigma
                # Var = Measurement.var.squeeze()
                
                # first_term= Measurement.N*torch.log(torch.tensor(2.0)*torch.pi)
                
                # second_term= torch.sum(torch.log(Var))
                
                # sse= torch.pow((torch.tensor(Measurement.MeasData)[0] - fx[ii,:]), torch.tensor(2.0)).squeeze()
                
                # third_term= torch.sum(sse/Var)
                
                # log_p[ii,0]= -0.5*(first_term+second_term+third_term)
                

    elif MCMCPar.lik==4: # join Be10 / Al26 inversion with 1 data point per data type
        Sigma=Measurement.Sigma
        N=np.ones((Measurement.N))
        of=np.zeros((fx.shape[0],1))
        log_p=np.zeros((fx.shape[0],1))
        for ii in range(0,fx.shape[0]):
            e=Measurement.MeasData-fx[ii,:]
            of[ii,0]=np.sqrt(np.sum(np.power(e,2.0))/e.shape[1])
            log_p_type=np.zeros((Measurement.N))
    
            for jj in range(0,Measurement.N):
                log_p_type[jj] = - ( N[jj] / 2.0) * np.log(2.0 * np.pi) - N[jj] * np.log( Sigma[jj] ) - 0.5 * np.power(Sigma[jj],-2.0) * np.sum( np.power(e[0,jj],2.0) )

            log_p[ii,0]=np.sum(log_p_type)
    
    if MCMCPar.lik==3.5: return rof, of, log_p
    else: return of, log_p

def GelmanRubin(Sequences,MCMCPar):
    """
    See:
    Gelman, A. and D.R. Rubin, 1992. 
    Inference from Iterative Simulation Using Multiple Sequences, 
    Statistical Science, Volume 7, Issue 4, 457-472.
    """
    
    n,nrp,m = Sequences.shape

    if n < 10:
        R_stat = -2 * np.ones((1,MCMCPar.n))
        
    else:
    
        meanSeq = np.mean(Sequences,axis=0)
        meanSeq = meanSeq.T
    
        # Variance between the sequence means 
        B = n * np.var(meanSeq,axis=0)
        
        # Variances of the various sequences
        varSeq=np.zeros((m,nrp))
        for zz in range(0,m):
            varSeq[zz,:] = np.var(Sequences[:,:,zz],axis=0)
        
        # Average of the within sequence variances
        W = np.mean(varSeq,axis=0)
        
        # Target variance
        sigma2 = ((n - 1)/np.float(n)) * W + (1.0/n) * B
        
        # R-statistic
        R_stat = np.sqrt((m + 1)/np.float(m) * sigma2 / W - (n-1)/np.float(m)/np.float(n))
    
    return R_stat
    
def DEStrategy(DEpairs,seq):
    
    # Determine which sequences to evolve with what DE strategy

    # Determine probability of selecting a given number of pairs
    p_pair = (1.0/DEpairs) * np.ones((1,DEpairs))
    p_pair = np.cumsum(p_pair)
    p_pair = np.concatenate((np.zeros((1)),p_pair),axis=0)
    
    DEversion=np.zeros((seq),dtype=np.int32)
    Z = np.random.rand(seq)
    # Select number of pairs
    for qq in range(0,seq):
        z = np.where(p_pair<=Z[qq])
        DEversion[qq] = z[0][-1]
            
    return DEversion
        
def BoundaryHandling(x,lb,ub,BoundHandling,lb_tot_eros=None,ub_tot_eros=None): 
    
    m,n=np.shape(x)
    
    # Replicate lb and ub
    minn = np.tile(lb,(m,1))
    maxn = np.tile(ub,(m,1))
    
    ii_low = np.argwhere(x<minn)
    ii_up = np.argwhere(x>maxn)
         
        
    if BoundHandling=='Reflect':
       
         # reflect in minn
        x[ii_low[:,0],ii_low[:,1]]=2 * minn[ii_low[:,0],ii_low[:,1]] - x[ii_low[:,0],ii_low[:,1]]      

         # reflect in maxn
        x[ii_up[:,0],ii_up[:,1]]=2 * maxn[ii_up[:,0],ii_up[:,1]] - x[ii_up[:,0],ii_up[:,1]] 
         
    if BoundHandling=='Bound':
         # set lower values to minn
        x[ii_low[:,0],ii_low[:,1]]= minn[ii_low[:,0],ii_low[:,1]] 
    
        # set upper values to maxn
        x[ii_up[:,0],ii_up[:,1]]= maxn[ii_up[:,0],ii_up[:,1]] 
        
    if BoundHandling=='Fold':
         # Fold parameter space lower values
        x[ii_low[:,0],ii_low[:,1]] = maxn[ii_low[:,0],ii_low[:,1]] - ( minn[ii_low[:,0],ii_low[:,1]] - x[ii_low[:,0],ii_low[:,1]]  )
    
        # Fold parameter space upper values
        x[ii_up[:,0],ii_up[:,1]] = minn[ii_up[:,0],ii_up[:,1]] + ( x[ii_up[:,0],ii_up[:,1]]  - maxn[ii_up[:,0],ii_up[:,1]] )
             
    # Now double check in case elements are still out of bound -- this is
    # theoretically possible if values are very small or large              
    ii_low = np.argwhere(x<minn)
    ii_up = np.argwhere(x>maxn)
    
    if ii_low.size > 0:
       
        x[ii_low[:,0],ii_low[:,1]] = minn[ii_low[:,0],ii_low[:,1]] + np.random.rand(ii_low.shape[0]) * (maxn[ii_low[:,0],ii_low[:,1]] - minn[ii_low[:,0],ii_low[:,1]])
   
    if ii_up.size > 0:
      
        x[ii_up[:,0],ii_up[:,1]] = minn[ii_up[:,0],ii_up[:,1]] + np.random.rand(ii_up.shape[0]) * (maxn[ii_up[:,0],ii_up[:,1]] - minn[ii_up[:,0],ii_up[:,1]])
   
    return x
    
def DreamzsProp(xold,Zoff,CR,MCMCPar,Update):
    
    # Determine how many pairs to use for each jump in each chain
    DEversion = DEStrategy(MCMCPar.DEpairs,MCMCPar.seq)

    # Generate uniform random numbers for each chain to determine which dimension to update
    D = np.random.rand(MCMCPar.seq,MCMCPar.n)

    # Generate noise to ensure ergodicity for each individual chain
    noise_x = MCMCPar.eps * (2 * np.random.rand(MCMCPar.seq,MCMCPar.n) - 1)

    # Initialize the delta update to zero
    delta_x = np.zeros((MCMCPar.seq,MCMCPar.n))

    if Update=='Parallel_Direction_Update':

        # Define which points of Zoff to use to generate jumps
        rr=np.zeros((MCMCPar.seq,4),dtype=np.int32())
        rr[0,0] = 0; rr[0,1] = rr[0,0] + DEversion[0]
        rr[0,2] = rr[0,1] +1 ; rr[0,3] = rr[0,2] + DEversion[0]
        # Do this for each chain
        for qq in range(1,MCMCPar.seq):
            # Define rr to be used for population evolution
            rr[qq,0] = rr[qq-1,3] + 1; rr[qq,1] = rr[qq,0] + DEversion[qq] 
            rr[qq,2] = rr[qq,1] + 1; rr[qq,3] = rr[qq,2] + DEversion[qq] 
 

        # Each chain evolves using information from other chains to create offspring
        for qq in range(0,MCMCPar.seq):

            # ------------ WHICH DIMENSIONS TO UPDATE? USE CROSSOVER ----------
            i = np.where(D[qq,:] > (1-CR[qq]))
            
            # Update at least one dimension
            if not i:
                i=np.random.permutation(MCMCPar.n)
                i=np.zeros((1,1),dtype=np.int32)+i[0]
       
              
        # -----------------------------------------------------------------

            # Select the appropriate JumpRate and create a jump
            if (np.random.rand(1) < (1 - MCMCPar.pJumpRate_one)):
                
                # Select the JumpRate (dependent of NrDim and number of pairs)
                NrDim = len(i[0])
                JumpRate = MCMCPar.Table_JumpRate[NrDim-1,DEversion[qq]]*MCMCPar.jr_scale
               
                # Produce the difference of the pairs used for population evolution
                if MCMCPar.DEpairs==1:
                    delta = Zoff[rr[qq,0],:]- Zoff[rr[qq,2],:]
                else:
                    # The number of pairs has been randomly chosen between 1 and DEpairs
                    delta = np.sum(Zoff[rr[qq,0]:rr[qq,1]+1,:]- Zoff[rr[qq,2]:rr[qq,3]+1,:],axis=0)
                
                # Then fill update the dimension
                delta_x[qq,i] = (1 + noise_x[qq,i]) * JumpRate*delta[i]
            else:
                # Set the JumpRate to 1 and overwrite CR and DEversion
                JumpRate = 1; CR[qq] = -1
                # Compute delta from one pair
                delta = Zoff[rr[qq,0],:] - Zoff[rr[qq,3],:]
                # Now jumprate to facilitate jumping from one mode to the other in all dimensions
                delta_x[qq,:] = JumpRate * delta
       

    if Update=='Snooker_Update':
                 
        # Determine the number of rows of Zoff
        NZoff = np.int64(Zoff.shape[0])
        
        # Define rr and z
        rr = np.arange(NZoff)
        rr = rr.reshape((2,np.int(rr.shape[0]/2)),order="F").T
        z=np.zeros((MCMCPar.seq,MCMCPar.n))
        # Define JumpRate -- uniform rand number between 1.2 and 2.2
        Gamma = 1.2 + np.random.rand(1)
        # Loop over the individual chains
        
        for qq in range(0,MCMCPar.seq):
            # Define which points of Zoff z_r1, z_r2
            zR1 = Zoff[rr[qq,0],:]; zR2 = Zoff[rr[qq,1],:]
            # Now select z from Zoff; z cannot be zR1 and zR2
            ss = np.arange(NZoff)+1; ss[rr[qq,0]] = 0; ss[rr[qq,1]] = 0; ss = ss[ss>0]; ss=ss-1
            t = np.random.permutation(NZoff-2)
            # Assign z
            z[qq,:] = Zoff[ss[t[0]],:]
                            
            # Define projection vector x(qq) - z
            F = xold[qq,0:MCMCPar.n] - z[qq,:]; Ds = np.maximum(np.dot(F,F.T),1e-300)
            # Orthogonally project of zR1 and zR2 onto F
            zP = F*np.sum((zR1-zR2)*F)/Ds
            
            # And define the jump
            delta_x[qq,:] = Gamma * zP
            # Update CR because we only consider full dimensional updates
            CR[qq] = 1
          
    # Now propose new x
    xnew = xold + delta_x
    
    # Define alfa_s
    if Update == 'Snooker_Update':
       
        # Determine Euclidean distance
        ratio=np.sum(np.power((xnew-z),2),axis=1)/np.sum(np.power((xold-z),2),axis=1)
        alfa_s = np.power(ratio,(MCMCPar.n-1)/2.0).reshape((MCMCPar.seq,1))
            
    else:
        alfa_s = np.ones((MCMCPar.seq,1))

    # Do boundary handling -- what to do when points fall outside bound
    if not(MCMCPar.BoundHandling==None):
        
#        if not(MCMCPar.BoundHandling=='CRN'):
        xnew = BoundaryHandling(xnew,MCMCPar.lb,MCMCPar.ub,MCMCPar.BoundHandling)
#        else:
#            xnew = BoundaryHandling(xnew,MCMCPar.lb,MCMCPar.ub,MCMCPar.BoundHandling,MCMCPar.lb_tot_eros,MCMCPar.ub_tot_eros)

    
    return xnew, CR ,alfa_s

def Metrop(MCMCPar,xnew,log_p_xnew,xold,log_p_xold,alfa_s,Extra):
    
    accept = np.zeros((MCMCPar.seq))
   
    # Calculate the Metropolis ratio based on the log-likelihoods
    alfa = np.exp(log_p_xnew.flatten() - log_p_xold)

    if MCMCPar.Prior=='StandardNormal': # Standard normal prior
        log_prior_new=np.zeros((MCMCPar.seq))
        log_prior_old=np.zeros((MCMCPar.seq))
            
        for zz in range(0,MCMCPar.seq):
            # Compute (standard normal) prior log density of proposal
            log_prior_new[zz] = -0.5 * reduce(np.dot,[xnew[zz,:],xnew[zz,:].T])     
             
            # Compute (standard normal) prior log density of current location
            log_prior_old[zz] = -0.5 * reduce(np.dot,[xold[zz,:],xold[zz,:].T])

        # Take the ratio
        alfa_pr = np.exp(log_prior_new - log_prior_old)
        # Now update alpha value with prior
        alfa = alfa*alfa_pr 

    # Modify for snooker update, if any
    alfa = alfa * alfa_s.flatten()
    
    # Generate random numbers
    Z = np.random.rand(MCMCPar.seq)
     
    # Find which alfa's are greater than Z
    idx = np.where(alfa > Z)[0]
    
    # And indicate that these chains have been accepted
    accept[idx]=1
    
    return accept

def Dreamzs_finalize(MCMCPar,Sequences,Z,outDiag,fx,iteration,iloc,pCR,m_z,m_func):
    
    # Start with CR
    outDiag.CR = outDiag.CR[0:iteration-1,0:pCR.shape[1]+1]
    # Then R_stat
    outDiag.R_stat = outDiag.R_stat[0:iteration-1,0:MCMCPar.n+1]
    # Then AR 
    outDiag.AR = outDiag.AR[0:iteration-1,0:2] 
    # Adjust last value (due to possible sudden end of for loop)

    # Then Sequences
    Sequences = Sequences[0:iloc+1,0:MCMCPar.n+2,0:MCMCPar.seq]
    
    # Then the archive Z
    Z = Z[0:m_z,0:MCMCPar.n+2]


    if MCMCPar.savemodout==True:
       # remove zeros
       fx = fx[:,0:m_func]
    
    return Sequences,Z, outDiag, fx
    
def Genparset(Sequences):
    # Generates a 2D matrix ParSet from 3D array Sequences

    # Determine how many elements in Sequences
    NrX,NrY,NrZ = Sequences.shape 

    # Initalize ParSet
    ParSet = np.zeros((NrX*NrZ,NrY))

    # If save in memory -> No -- ParSet is empty
    if not(NrX == 0):
        # ParSet derived from all sequences
        tt=0
        for qq in range(0,NrX):
            for kk in range(0,NrZ):
                ParSet[tt,:]=Sequences[qq,:,kk]
                tt=tt+1
    return ParSet

def forward_parallel(forward_process,X,n,n_jobs,extra_par): 
    
    n_row=X.shape[0]
    
    parallelizer = Parallel(n_jobs=n_jobs)
    
    tasks_iterator = ( delayed(forward_process)(X_row,n,extra_par) 
                      for X_row in np.split(X,n_row))
         
    result = parallelizer( tasks_iterator )
    # Merging the output of the jobs
    return np.vstack(result)
      
def RunFoward(X,MCMCPar,Measurement,ModelName,Extra,DNN=None):
    
    
    n=Measurement.N
    n_jobs=Extra.n_jobs
    
    if ModelName=='theoretical_case_mvn':
        extra_par = Extra.invC
        
    elif ModelName=='theoretical_case_bimodal_mvn':
        extra_par = []
        extra_par.append(Extra.mu1)
        extra_par.append(Extra.cov1)
        extra_par.append(Extra.mu2)
        extra_par.append(Extra.cov2)
        
    elif ModelName=='linear_gpr_tomo':
        extra_par=[]
        # Generate realizations
        zs=np.zeros((MCMCPar.seq,DNN.nz,DNN.zx,DNN.zy))
        for i in range(0,MCMCPar.seq):
            zs[i,:]=X[i,:].reshape((DNN.nz,DNN.zx,DNN.zy))
        zs = torch.from_numpy(zs).float()
        if DNN.cuda:
            zs = zs.cuda()
            
        m = DNN.netG(zs).cpu().numpy()
        # Crop model and get rid of unecessary dimension
        m = m[:,0,2:127,3:63]
  
        
        m = (m + 1) * 0.5  # Convert from [-1,1] to [0,1]
        
        if DNN.filtering:  # always False herein
            for ii in range(m.shape[0]):
                m[ii] = medfilt(m[ii], kernel_size=(3, 3))
    
        if DNN.threshold: # categorical case
            m[m < 0.5] = 0
            m[m >= 0.5] = 1
    
            m[m==0]=0.08 # m/ns
            m[m==1]=0.06 # m/ns
        else: # continuous case
            m = 1 - m
            m= 0.06 + m*0.02
        
        # Linear operator parameters
        extra_par.append(Extra.G)
        X=m   
        
    elif ModelName=='nonlinear_pygimli': 
        extra_par=[]
        extra_par.append(Extra.bh_spacing)
        extra_par.append(Extra.bh_length)
        extra_par.append(Extra.sensor_spacing)

        # Generate realizations
        if MCMCPar.DGM=='SGAN':
            zs=np.zeros(((MCMCPar.seq,DNN.nz)+DNN.dims))
            for i in range(0,MCMCPar.seq):
                zs[i,:]=X[i,0:MCMCPar.n].reshape(((DNN.nz,)+DNN.dims))
        elif MCMCPar.DGM=='VAE':
            zs=np.zeros(((MCMCPar.seq,)+DNN.dims))
            for i in range(0,MCMCPar.seq):
                zs[i,:]=X[i,0:MCMCPar.n].reshape(DNN.dims)            
        zs = torch.from_numpy(zs).float()
        if DNN.cuda:
            zs = zs.cuda()
            
        m = DNN.netG(zs).cpu().numpy()

        if MCMCPar.DGM == 'SGAN':
            m = 0.5*(m+1)  # normalize generated model to [0,1]
        
        # assign velocity to the model (0.06 and 0.08 m/ns)
        if DNN.filtering:  # always False herein
            for ii in range(m.shape[0]):
                m[ii] = medfilt(m[ii], kernel_size=(3, 3))
    
        if DNN.threshold: # categorical case
            m[m < 0.5] = 0
            m[m >= 0.5] = 1
    
            m[m==0]=0.08 # m/ns
            m[m==1]=0.06 # m/ns
        else: # continuous case
            m = 1 - m
            m= 0.06 + m*0.02
            
        X = m[:,0]
        
    elif ModelName=='fullstack_seismic': # !!!! adapted for fullstack inversion
        extra_par=[]
        extra_par.append(Extra.w)

        # Generate realizations
        if MCMCPar.DGM=='GAN':
            zs=np.zeros(((MCMCPar.seq,DNN.nz)+DNN.dims))
            for i in range(0,MCMCPar.seq):
                zs[i,:]=X[i,0:MCMCPar.n].reshape(((DNN.nz,)+DNN.dims))
        elif MCMCPar.DGM=='VAE':
            zs=np.zeros(((MCMCPar.seq,)+DNN.dims))
            for i in range(0,MCMCPar.seq):
                zs[i,:]=X[i,0:MCMCPar.n].reshape(DNN.dims)            
        zs = torch.from_numpy(zs).float()
        if DNN.cuda:
            zs = zs.cuda()
        
        f,ip = DNN.netG(zs)

        if MCMCPar.DGM == 'GAN':
            f = 0.5*(f+1)  # normalize generated model to [0,1]
        
        X = (Extra.Ip_lmts[1]-Extra.Ip_lmts[0])*ip+Extra.Ip_lmts[0]
    
       
    else:
        extra_par=None
    
    forward_process=getattr(sys.modules[__name__], ModelName)
    
    if MCMCPar.DoParallel:
    
        start_time = time.time()
        
        fx=forward_parallel(forward_process,X,n,n_jobs,extra_par)

        end_time = time.time()
        elapsed_time = end_time - start_time
    
        if not(ModelName[0:4]=='theo'):
            pass
            #print("Parallel forward calls done in %5.4f seconds." % (elapsed_time))
    else:
        fx=np.zeros((X.shape[0],n))
        
        start_time = time.time()
        if ModelName=='fullstack_seismic': 
            fx= forward_process(X,extra_par[0]).detach().cpu().numpy()

                        
        elif not(ModelName[0:4]=='theo'): 
            for qq in range(0,X.shape[0]):
                fx[qq,:]=forward_process(X[qq],n,extra_par)
        
        else:
            for qq in range(0,X.shape[0]):
                fx[qq,:]=forward_process(X[qq,:],n,extra_par)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
         
        if not(ModelName[0:4]=='theo'):
            #print("Sequential forward calls done in %5.4f seconds." % (elapsed_time))
            pass
    return fx
    
def linear_gpr_tomo(m,n,par):

    G=par[0]
    s=1/m # from velocity field to slowness field
    sim=G@s.flatten(order='F')

    return sim

def nonlinear_pygimli(m,n,par):
    resp = pygimli_fw(m, par[0], par[1], par[2])
    return resp

def fullstack_seismic(m,w):
    r= reflectivity_ip(m)
    w = w[None,:]
    w= torch.moveaxis(w, source=-1, destination=1)
    padding= w.shape[-2]//2,0
    
    return conv2d(r, w, padding=padding)

def theoretical_case_mvn(X, n, icov):

    fx=np.zeros((1,n))
    # Calculate the log density of zero mean correlated mvn distribution
    fx[0,:n] = -0.5*X.dot(icov).dot(X.T)

    return fx
    
def theoretical_case_bimodal_mvn(X, n, par):
    
    fx=np.zeros((1,n))
    fx[0,:n] = (1.0/3)*multivariate_normal.pdf(X, mean=par[0], cov=par[1])+(2.0/3)*multivariate_normal.pdf(X, mean=par[2], cov=par[3])
    
    return fx

def facies(x):
    return x