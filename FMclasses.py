import torch
from torch import nn
from torch.nn.functional import conv2d
import numpy as np
import subprocess 
from Gslib import Gslib

class ForwardModeling(nn.Module):
    def __init__(self, args):
        super(ForwardModeling, self).__init__()
        
        self.device= args.device
        self.type_of_FM= args.type_of_FM
        self.syn_seismic= torch.zeros((args.batch_size, 1, args.nz, args.nx)).to(self.device)
        self.wavelets= None
        
    def load_wavelet(self,args):
        wavelets = np.genfromtxt(args.project_path+args.in_folder+args.wavelet_file)
        wavelets= wavelets*args.W_weight

        wavelets = np.expand_dims(wavelets, 0) # add bacth [B x H x W x C]
        if wavelets.ndim==2: 
            wavelets= np.expand_dims(wavelets, 0)
            wavelets= np.expand_dims(wavelets, -1)
            
        self.wavelets = torch.from_numpy(wavelets).double().to(self.device)
        
        k = self.wavelets.shape[-2]
        self.padding = (k//2,0)

        self.angles = [0]
    
    def reflectivity_ip(self, ip):
        ip = torch.cat((ip, ip[:,:,[-1],:]), dim=2) # repeats last element
        ip_d =  ip[:, :, 1:, :] - ip[:, :, :-1, :]
        ip_a = (ip[:, :, 1:, :] + ip[:, :, :-1, :])    
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

    def reflectivity_aki(self, x, angles=None):
        x = torch.cat((x, x[:,:,[-1],:]), dim=2) # repeats last element

        vp1 = x[:, :, :-1, 0]
        vs1 = x[:, :, :-1, 1]
        rho1 = x[:, :, :-1, 2]
        vp2 = x[:, :, 1:, 0]
        vs2 = x[:, :, 1:, 1]
        rho2 = x[:, :, 1:, 2]
        
        if angles==None: angles= self.angels
        dim = x.shape
        rc = torch.zeros((dim[0], dim[1], dim[2], len(angles)))
        for i, angle in enumerate(angles):
            rc[...,i] = self.akirichards(vp1, vs1, rho1, vp2, vs2, rho2, angle)
        return rc

    def forward(self, el_model, angles=None):
        """typ=0 : ip for post-stack seismic data
           typ=1 : vp vs den aki partial-stack seismic data
        """
        if self.type_of_FM=='fullstack': rc = self.reflectivity_ip(el_model)
        elif self.type_of_FM=='partialstack': rc = self.reflectivity_aki(el_model,angles)
        seismic = conv2d(rc.double().to(self.device), self.wavelets, padding=self.padding)

        return seismic
    
    
    def calc_synthetic(self, el_models, angles=None):
        
        self.syn_seismic = torch.zeros_like(el_models)
        for i in range(self.syn_seismic.shape[0]):
            self.syn_seismic[i]= self.forward(el_models[i,None,:], angles)
        
        return self.syn_seismic
    
    def check_seis_distance(self, args):
        r= self.real_seismic
        s= self.syn_seismic
        if r.shape[0]!=s.shape[0]: r= torch.tile(r, (s.shape[0],1,1,1))
        r= r + torch.randn_like(r)/1000
        s= s + torch.randn_like(s)/1000
        
        if args.type_of_corr=='similarity2':
            num= torch.abs(torch.subtract(s, r))
            den= torch.add(torch.abs(r), torch.abs(s))
            
            simil = (num/den)
            g= torch.mean(simil.reshape(-1,args.nx*args.nz))
        
    
        elif args.type_of_corr=='RMSE':
            sqrde= torch.nn.MSELoss(reduction='none')(s,r).reshape(-1,args.nx*args.nz)
            g= torch.sum(sqrde, axis=-1)/(args.nx*args.nz)
            if args.type_of_corr=='RMSE': g= g**1/2

        else:
            g= torch.nn.MSELoss()(s,r)
            
        return g
    
        
class ElasticModels():
    def __init__(self, args, real_fac_model=None, ipmin=None, ipmax=None,ipzones=None):
           
        if 'in_folder' not in args:
            args.in_folder=''
        if 'type_of_FM' not in args:
            args.type_of_FM = ''
            
        self.inf= f'{args.workdir}/{args.in_folder}/'
        self.ouf= f'{args.workdir}/{args.outdir}/'
        self.nx= args.nx
        self.nz= args.nz
        self.var_N_str= args.var_N_str if 'var_N_str' in args else 0
        self.var_nugget= args.var_nugget if 'var_nugget' in args else 0
        self.var_type= args.var_type  if 'var_type' in args else 0
        self.var_ang= args.var_ang  if 'var_ang' in args else 0
        self.var_range= args.var_range   if 'var_range' in args else 0
        self.null_val= args.null_val if 'null_var' in args else -9999.99
        
        if args.type_of_FM=='fullstack':
            self.simulations= torch.zeros((args.batch_size, 1, args.nz, args.nx))
        else:
            pass

        try:
            self.ipmin= ipmin
            self.ipmax= ipmax
            self.ipzones= ipzones
        except: 
            raise TypeError('No Ip values bounds per facies provided')

        if real_fac_model!=None: 
            if args.ip_type==0:
                self.real_model= self.det_Ip(real_fac_model)
            elif args.ip_type==1:
                real_fac_model = (real_fac_model+1)*0.5
                self.real_model= self.run_dss(real_fac_model)[0,None,:]
        
            
    def det_Ip(self, facies_model):
        facies_model= (facies_model+1)*0.5
        #ip= torch.zeros_like(facies_model)

        #ip[facies_model<0.5]=self.ipmax
        #ip[facies_model>=0.5]=self.ipmin
        #return ip
        return (self.ipmin-self.ipmax)*facies_model+self.ipmax
    
    
    def writeallfac_dss(self, facies_mod):
        facies_mod= np.round(facies_mod.reshape(-1,facies_mod.shape[-2]*facies_mod.shape[-1]))
        for f in range(facies_mod.shape[0]):
            with open(self.ouf+f'/dss/Facies_model_{f+1}.out','w') as fid:
                fid.write('Facies\n')
                fid.write('1\n')
                fid.write('Facies\n')
                fid.write('\n'.join(facies_mod[f].astype(int).astype(str).tolist()))
        return None
        
    def write_parfile(self,s_f,typ,nsim):
        #writes the parfile for DSS for each facies model    
        text=[]
        text.append(f'[ZONES]\nZONESFILE = {self.ouf}/dss/Facies_model_{s_f+1}.out  # File with zones\nNZONES={len(self.ipzones)}  # Number of zones\n\n')
        for fac in range(len(self.ipzones)):
            text.append(f'[HARDDATA{fac+1}]\nDATAFILE = {self.inf}/Ip_zone{fac}.gslib  # Hard Data file\n')
            text.append('COLUMNS = 4\nXCOLUMN = 1\nYCOLUMN = 2\nZCOLUMN = 3\nVARCOLUMN = 4\nWTCOLUMN = 0\n')
            text.append(f'MINVAL = {self.ipzones[fac][0]}  # Minimun threshold value\nMAXVAL = {self.ipzones[fac][1]}  # Minimun threshold value\n')
            text.append('USETRANS = 1\nTRANSFILE = Cluster.trn  #Transformation file\n\n')
        text.append(f'[HARDDATA]\nZMIN = {self.ipmin}  # Minimum allowable data value\nZMAX = {self.ipmax}  # Maximum allowable data value\nLTAIL = 1\nLTPAR = {self.ipmin}\nUTAIL = 1\nUTPAR = {self.ipmax}\n\n')
        text.append(f'[SIMULATION]\nOUTFILE = {self.ouf}/dss/ip_real  # Filename of the resultant simulations\nNSIMS = {nsim}  # Number of Simulations to generate \nNTRY = 10\nAVGCORR = 1\nVARCORR = 1\n\n')
        text.append(f'[GRID]\nNX = {self.nx}\nNY = 1\nNZ = {self.nz}\nORIGX = 1\nORIGY = 1\nORIGZ = 1\nSIZEX = 1\nSIZEY = 1\nSIZEZ = 1\n\n')
        text.append(f'[GENERAL]\nNULLVAL = {self.null_val} \nSEED = 6906\nUSEHEADERS = 1\nFILETYPE = GEOEAS\n\n')
        text.append(f'[SEARCH]\nNDMIN = 1\nNDMAX = 32\nNODMAX = 12\nSSTRAT = 1\nMULTS = 0\nNMULTS = 1\nNOCT = 0\nRADIUS1 = {self.nx}\nRADIUS2 = 1\nRADIUS3 = {self.nz}\nSANG1 = 0\nSANG2 = 0\nSANG3 = 0\n\n')
        text.append('[KRIGING]\nKTYPE = 0  # Kriging type: 0=simple,1=ordinary,2=simple with locally varying mean, 3=external drif, 4=collo-cokrig global CC,5=local CC (KTYPE)\n')
        text.append('COLOCORR = 0.75\nSOFTFILE = No File\nLVMFILE = No File\nNVARIL = 1\nICOLLVM = 1\nCCFILE = No File\nRESCALE = 1\n\n')
        
        for fac in range(len(self.ipzones)):
            text.append(f'[VARIOGRAMZ{fac+1}]\nNSTRUCT = {self.var_N_str[fac]}  # Number of semivariograms structures\nNUGGET = {self.var_nugget[fac]}  # Nugget constant\n\n')
            for struct in range(self.var_N_str[fac]):
                text.append(f'[VARIOGRAMZ{fac+1}S{struct+1}]\nTYPE = {self.var_type[fac][struct]}\nCOV = 1\n')
                text.append(f'ANG1 = {self.var_ang[fac][0]}\nANG2 = 0\nANG3 = {self.var_ang[fac][1]}\n')
                text.append(f'AA = {self.var_range[fac][struct][0]}\nAA1 = 1\nAA2 = {self.var_range[fac][struct][1]}\n\n')
            text.append(f'[BIHIST{fac+1}]\nUSEBIHIST = 0\nBIHISTFILE = No File\nNCLASSES = 20\nAUXILIARYFILE = No File\n\n')
        text.append('[DEBUG]\nDBGLEVEL = 1\nDBGFILE = debug.dbg\n\n')
        text.append(f'[COVTAB]\nMAXCTX = {self.nx}\nMAXCTY = 1\nMAXCTZ = {self.nz}\n\n')
        text.append('[BLOCKS]\nUSEBLOCKS = 0\nBLOCKSFILE = NoFile\nMAXBLOCKS= 100\n\n[PSEUDOHARD]\nUSEPSEUDO = 0\nBLOCKSFILE = No File\nPSEUDOCORR = 0\n')
                
        text= ''.join(text)
        
        with open(self.inf+'/ssdir.par', 'w') as ssdir:
            ssdir.write(text)
            

    def run_dss(self, facies_mod, typ='unc', nsim=1):
        self.simulations= torch.zeros((facies_mod.shape[0], 1, facies_mod.shape[2], facies_mod.shape[3]))
        if facies_mod.min() < 0: facies_mod= (facies_mod.detach().cpu().numpy()+1)*0.5
        else: facies_mod= facies_mod.detach().cpu().numpy()
        self.writeallfac_dss(facies_mod)

        for s_f in range(facies_mod.shape[0]):
            self.write_parfile(s_f,typ,nsim)
            
            subprocess.run(args=[f'{self.inf}dss_unprotected', f'{self.inf}ssdir.par'], stdout=subprocess.DEVNULL)
            
            ssimss= np.zeros((nsim,1,self.nz,self.nx))
            for ssi in range (0,nsim):
                ssimss[ssi]= np.reshape(Gslib().Gslib_read(f'{self.ouf}/dss/ip_real_{ssi+1}.out').data.values.squeeze(),
                              (self.simulations.shape[-3], self.simulations.shape[-2], self.simulations.shape[-1]))

            self.simulations[s_f]= torch.mean(torch.from_numpy(ssimss), axis=0)
            
            # self.simulations[s_f]=torch.from_numpy(np.reshape(Gslib().Gslib_read(f'{self.ouf}/dss/ip_real_1.out').data.values.squeeze(),
            #                   (self.simulations.shape[-3], self.simulations.shape[-2], self.simulations.shape[-1])))

        return self.simulations
           

    def prerun_DSS(self, args, facies_mods, typ='unc', nsim=1):
        # self.simulations= torch.zeros((facies_mod.shape[0], 1, facies_mod.shape[2], facies_mod.shape[3]))
        # if facies_mod.min() < 0: facies_mod= (facies_mod.detach().cpu().numpy()+1)*0.5
        # else: facies_mod= facies_mod.detach().cpu().numpy()
        
        
        for s_f in range(facies_mods.shape[0]):
            self.write_parfile(s_f,typ,nsim)
            
            subprocess.run(args=[f'{self.inf}DSS.C.64.exe', f'{self.inf}ssdir.par'], stdout=subprocess.DEVNULL)
            
            ssimss= np.zeros((nsim,1,self.nz,self.nx))
            for ssi in range (0,nsim):
                ssimss[ssi]= np.reshape(Gslib().Gslib_read(f'{self.ouf}/dss/ip_real_{ssi+1}.out').data.values.squeeze(),
                              (self.simulations.shape[-3], self.simulations.shape[-2], self.simulations.shape[-1]))

            self.simulations[s_f]= torch.mean(torch.from_numpy(ssimss), axis=0)
            
            # self.simulations[s_f]=torch.from_numpy(np.reshape(Gslib().Gslib_read(f'{self.ouf}/dss/ip_real_1.out').data.values.squeeze(),
            #                   (self.simulations.shape[-3], self.simulations.shape[-2], self.simulations.shape[-1])))
        
        self.ouf= f'{args.project_path}/{args.out_folder}/'
        return self.simulations
