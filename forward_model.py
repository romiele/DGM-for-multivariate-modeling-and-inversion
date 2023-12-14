import torch
from torch import nn
from torch.nn.functional import conv2d
import numpy as np
import os
from dataclasses import dataclass

@dataclass
class setup(nn.Module):
    #storing forward model parameters 
    # zinit: np.ndarray = np.zeros(1)
    sigma: float = 0.1 
    gen: nn.Module = None
    fwd: str = 'fullstack'
    A: np.ndarray = np.zeros(1)
    tt: str = None
    d: np.ndarray = np.zeros(1)
    truemodel: np.ndarray = np.zeros(1)
    def __init__(self, wavelet: np.array, device='cpu'):    
        '''
        wavelets: tensor of dimensions [H, W, C (#angles)]
        '''
        super(setup, self).__init__()
        
        wavelet = np.expand_dims(wavelet.detach().cpu(), 0) # add bacth [B x H x W x C]
        wavelet = np.moveaxis(wavelet, source=-1, destination=1)
        self.wavelet = torch.from_numpy(wavelet).double().to(device)    
        
        k = self.wavelet.shape[-2]
        self.padding = (k//2,0)

        self.angles = [0]
        


def reflectivity_ip(ip):
    ip = torch.cat((ip, ip[:,:,[-1],:]), dim=2) # repeats last element
    ip_d =  ip[:, :, 1:, :] - ip[:, :, :-1, :]
    ip_a = (ip[:, :, 1:, :] + ip[:, :, :-1, :]) / 2    
    return ip_d / ip_a        

def akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    theta2 = torch.arcsin(vp2/vp1*torch.sin(theta1))
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    meantheta = (theta1+theta2) / 2.0
    rho = (rho1+rho2) / 2.0
    vp = (vp1+vp2) / 2.0
    vs = (vs1+vs2) / 2.0

    # Compute the coefficients
    w = 0.5 * drho/rho
    x = 2 * (vs/vp1)**2 * drho/rho
    y = 0.5 * (dvp/vp)
    z = 4 * (vs/vp1)**2 * (dvs/vs)

    # Compute the terms
    term1 = w
    term2 = -1 * x * torch.sin(theta1)**2
    term3 = y / torch.cos(meantheta)**2
    term4 = -1 * z * torch.sin(theta1)**2

    return term1 + term2 + term3 + term4

def reflectivity_aki(x, angles):
    x = torch.cat((x, x[:,:,[-1],:]), dim=2) # repeats last element

    vp1 = x[:, :, :-1, 0]
    vs1 = x[:, :, :-1, 1]
    rho1 = x[:, :, :-1, 2]
    vp2 = x[:, :, 1:, 0]
    vs2 = x[:, :, 1:, 1]
    rho2 = x[:, :, 1:, 2]

    dim = x.shape
    rc = torch.zeros((dim[0], dim[1], dim[2], len(angles)))
    for i, angle in enumerate(angles):
        rc[...,i] = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, angle)
    return rc

def forward(x, fpars):
    """typ=0 : ip for post-stack seismic data
       typ=1 : vp vs den aki partial-stack seismic data
    """
    if fpars.fwd=='fullstack': rc = reflectivity_ip(x)
    if fpars.fwd=='partialstack': rc = reflectivity_aki(x)
    synth = conv2d(rc.double(), fpars.wavelet, padding=fpars.padding)

    return synth

def fullstack_solver_simple(ip_true, FWpars, ncase=0):
    #ip_true = x_true.clone() # shape: [1, 1, ny, nx]
    
     #continuous function 8000 (sands) and 11000 (shale)
    # Convert to fixed ip
    # ip_true[x_true<0.5]=11000
    # ip_true[x_true>=0.5]=8000
    d= forward(ip_true, FWpars)
    
    if ncase == 'noise':
        # add synthetic noise torch.randn_like(m_seis)*(m_seis*setup.relative_error+setup.c_sigma)
        noise_lvl = FWpars.relative_error*torch.abs(d.detach().clone())+FWpars.c_sigma
        nnoise = noise_lvl*torch.randn_like(d)
        # nnoise = np.load(os.getcwd()+'/test_models/noiserealz.npy') # ns
        d = d + nnoise # noise_lvl scales the noise.
        # print(noise_lvl*nnoise[:10]) # ns
    return d
    