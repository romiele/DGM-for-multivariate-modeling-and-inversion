"""
New plot results for seismic data
"""

import os
import numpy as np
import csv
# import torch and pyro modules
import torch
import pyro
import pyro.distributions as dist
# plotting libraries
import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.rc('legend', framealpha = 1)
# plt.rc('text', usetex=True)
from matplotlib.gridspec import GridSpec
import seaborn as sns
# import SeabornFig2Grid as sfg
import forward_model as fm 
from sklearn.metrics import mean_squared_error as mse

os.chdir('C:/Users/romie/OneDrive - Universidade de Lisboa/4.Neural-Transport/2. Facies_Ip/')
from GAN import uGenerator_2layers
from VAE import VAE_2layers
from Neural_transport import model, Posterior

import dill as pickle
pickle.settings['recurse'] = False
import matplotlib 
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
matplotlib.rc('axes', labelsize=16)
matplotlib.rcParams['figure.facecolor'] = (1,1,1,0)
# workdir= 'C:/Users/roberto.miele/OneDrive - Universidade de Lisboa/4.Neural-Transport/2. Facies_Ip/VAE_mv5.pt_1/'

def main_plot(workdir):
    log_file= open(workdir+'/log_results.txt','w')
    
    print('Case study: ',workdir[-14:-2])
    log_file.write(f'Case study: {workdir[-14:-2]}\n')

    step = 'last'  # saved step number or 'last'
    num_samples = 500
    DoThreshold = 0  # threshold the images into [0,1]
    os.chdir(workdir)
    plot_latent = 1
    plot_MCMC = 1 if workdir[-10:-9]=='3' else 0
    plot_latent_step = 1
    

    with open(workdir+'/setup.pkl', 'rb') as f:

        if __name__== '__main__': 
            os.chdir(os.path.abspath(os.path.join(workdir, os.pardir)))
      
        setup = pickle.load(f)
        
    with open(workdir+'/loss.pkl', 'rb') as f:
        tmp = pickle.load(f)
    loss_vals = tmp[0]
    logS = tmp[1]
    num_pars = setup.total_event_size

    if setup.DGM == 'VAE':
        print('DGM is based on VAE')
        log_file.write('DGM is based on VAE')

        # load guide structure
        guide = torch.load(
            workdir+'/Guides/saved_IAF_guide_VAE.pt')
    elif setup.DGM == 'GAN':
        print('DGM is based on GAN')
        log_file.write('DGM is based on GAN')

        # load guide structure
        guide = torch.load(
            workdir+'/Guides/saved_IAF_guide_GAN.pt')
    else:
        print('no valid DGM')

    pyro.clear_param_store()
    # load pyro ParamStore
    # pyro.get_param_store().load(workdir+'NeuTra_paramstore')
    # load params to the guide
    guide.load_state_dict(torch.load(
        workdir+'saved_NT_params_step_'+str(step)+'.pt'))
    guide.eval()
    # check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if str(device) == 'cuda':
        cuda = True
    else:
        cuda = False

    pyro.set_rng_seed(0)

    base_dist = guide.get_base_dist()  # give samples from the base distribution
    posterior = guide.get_posterior()  # give samples from the posterior
    # guide.get_transform()  # detail the transform of the guide

    posterior_samples = posterior.rsample(
        sample_shape=torch.Size([num_samples])).to(device)

    base_samples = base_dist.rsample(sample_shape=torch.Size([num_samples]))
    
    m_f = setup.truemodel[0][0,0]
    m_ip = setup.truemodel[1][0,0]

    m_seis= fm.fullstack_solver_simple(m_ip[None,None,:], setup, 'noise')[0,0]
    if setup.DGM == 'GAN':
        post_img_f,post_img_ip = setup.gen(posterior_samples.detach().cpu())
        post_img_f = (post_img_f + 1) * 0.5

    elif setup.DGM == 'VAE':
        post_img_f, post_img_ip = setup.gen(posterior_samples.detach().cpu())
    
    post_img_ip= (setup.ipmax-setup.ipmin)*post_img_ip.clone()+setup.ipmin

    post_img_seis= fm.fullstack_solver_simple(post_img_ip, setup, 'noise').detach().numpy().squeeze()
    
    post_f_std = np.std(post_img_f[:,0].detach().cpu().numpy(), axis = 0)

    post_img_f = post_img_f[:, 0].detach().cpu().numpy().squeeze()
    post_img_ip = post_img_ip[:, 0].detach().cpu().numpy().squeeze()
    mean_post_f = np.mean(post_img_f, axis=0).squeeze()
    mean_post_ip = np.mean(post_img_ip, axis=0).squeeze()
    mean_post_seis= np.mean(post_img_seis, axis=0).squeeze()
    var_ip = np.var(np.array(post_img_ip), axis=0)
    var_f = np.var(np.array(post_img_f), axis=0)
    
    mean_sigma= (torch.randn_like(m_seis)*(np.abs(m_seis)*setup.relative_error+setup.c_sigma)).std()
    
    relative_error= np.abs(mean_post_seis-m_seis.detach().numpy())/np.abs(m_seis.detach().numpy())
    
    if DoThreshold:
        threshold = 0.5
        # mean_post = (mean_post>threshold).float()
        # m = (m>threshold).float()
        mean_post_f[mean_post_f < threshold] = 0
        mean_post_f[mean_post_f >= threshold] = 1
        m_f[m_f < threshold] = 0
        m_f[m_f >= threshold] = 1

    r_misfit= torch.load(workdir +f'Data_e.pt').numpy()
    rmse_d = r_misfit[:,0]
    wrmse_d = r_misfit[:,1]

    #%% MCMC
    if plot_MCMC:    
        MCMC_work_dir = workdir
        results_folder = workdir
        with open(results_folder+'out_tmp'+'.pkl','rb') as f:
             tmp = pickle.load(f)
        Extra = tmp['Extra']
        OutDiag = tmp['OutDiag']
        Sequences = tmp['Sequences']
        MCMCPar = tmp['MCMCPar']
        MCMCVar =  tmp['MCMCVar']
        Measurement = tmp['Measurement']
        del tmp
    
        condition = Sequences[:,0,0] == 0
        index = condition.argmax() if condition.any() else len(Sequences[:,0,0])
    
        MCMC_misfit = Sequences[:index,-2,:]

    # %%
    RMSE_zs= np.zeros(500)
    for ij in range(500):
        RMSE_zs[ij] = mse(setup.zs.reshape(-1).detach().cpu().numpy(), 
                          posterior_samples.detach().cpu().numpy()[ij], squared=False)
                  # torch.mean(posterior_samples, axis=0).detach().cpu().numpy(), squared=False)
        
    print('Average RMSEz:', RMSE_zs.mean())
    
    RMSE_m_ip = mse(m_ip, mean_post_ip, squared=False)
    RMSE_m_f = mse(m_f, mean_post_f, squared=False)
    print('Average RMSE_ip:', RMSE_m_ip)
    print('Average RMSE_f:', RMSE_m_f)
    
    
    torch.sum(((m_seis - mean_post_seis)**2).reshape(-1,8000))/8000
    
    
    
    RMSE_d = mse(m_seis , mean_post_seis, squared=False)
    print('Average RMSE_d:', RMSE_d)    
    print('Average relative error: ', relative_error.mean())
    
    log_file.write(f'\nAverage RMSEz: {RMSE_zs.mean()}\n')
    log_file.write(f'Average RMSE_ip: {RMSE_m_ip}\n')
    log_file.write(f'Average RMSE_f: {RMSE_m_f}\n')
    log_file.write(f'Average RMSE_d: {RMSE_d}\n')
    log_file.write(f'Average relative error: {relative_error.mean()}\n')

    logS_zs = -posterior.log_prob(setup.zs.reshape(-1)).detach().numpy()
    print('logS --> '+str(logS_zs))
    log_file.write(f'logS (NT) --> {logS_zs}\n')
    
    
    # %%
    def RMSE_conv(misD):
        for i in range(len(misD)):
            if np.mean(misD[i:]).round(2)==np.mean(misD[len(misD)-int(len(misD)*0.1):len(misD)]).round(2) and np.mean(misD[i:]).round(2)<100: #np.std(misD[i:-1]).round(2)<0.01:
                index = i
                break
            else:
                index='## No convergence ##'
        return index
    
    conv_step = RMSE_conv(wrmse_d)
    print('Convergence is at step '+ str(conv_step))
    log_file.write(f'Convergence is at step: {conv_step}\n')

    # %%
    from skimage.metrics import structural_similarity as ssim
    SSIM_f, ssim_imgs_f = ssim(m_f.numpy(), mean_post_f, data_range=1.0, win_size=7, full=True)
    SSIM_ip, ssim_imgs_ip = ssim(m_ip.numpy(), mean_post_ip, data_range=1.0, win_size=7, full=True)

    print('SSIM value (Facies):', SSIM_f)
    print('SSIM value (Ip):', SSIM_ip )
    log_file.write(f'SSIM value (Facies): {SSIM_f}\n')
    log_file.write(f'SSIM value (Ip): {SSIM_ip}\n')
    
    # %%
    fig, axs = plt.subplots(3,1, sharey=True, sharex=True, figsize=(9.3,10), dpi=400)
    
    axs[0].set_title(f'{setup.outdir[4:7]}\n\n'+r'$\bf{d_{obs}}$', fontsize=20)
    axs[1].set_title(r'Facies', fontsize=20)
    axs[2].set_title(r'I$_P$', fontsize=20)
    ax1 = axs[0].imshow(m_seis.detach().cpu(), cmap='seismic', vmin=-800, vmax=800)
    ax2 = axs[1].imshow(m_f,cmap='gray', vmin=0,vmax=1)
    ax3 = axs[2].imshow(m_ip,cmap='jet', vmin=m_ip.min(), vmax=m_ip.max())
     
    for i in range(3):
        axs[i].set_ylabel('TWT [ms]', fontsize=18)
    axs[i].set_xlabel('CMP [m]', fontsize=18)
    
    for i in range(3): 
        bbox = axs[i].get_position()
        bbox.x0 -= 0.05; bbox.x1 -= 0.05
        axs[i].set_position(bbox)
        
    if setup.outdir[6] == '3':
        bbox= axs[0].get_position()
        cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.03, (bbox.y1-bbox.y0)])
        fig.colorbar(ax1, cax=cax)
        cax.set_ylabel('Amplitude',fontsize=18)
        cax.set_yticks([-800,0,800])
        
        bbox= axs[1].get_position()
        cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.03, (bbox.y1-bbox.y0)])
        fig.colorbar(ax2, cax=cax)
        cax.set_ylabel('Facies', labelpad=30,fontsize=18)
        cax.set_yticks([0,1])
    
        bbox= axs[2].get_position()
        cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.03, (bbox.y1-bbox.y0)])
        fig.colorbar(ax3, cax=cax)
        cax.set_ylabel(r'I$_P$ '+r'[m/s g/cm$^3$]',fontsize=18)
        cax.set_yticks([m_ip.min(),m_ip.max()])
        
    if setup.outdir[6] != '1':
        for i in range (3): 
            axs[i].yaxis.set_visible(False)
        
        
    plt.savefig(workdir+'ground_truth.png',dpi=400)    
    plt.close()
    
    # %% Plot grids NT
    fig, axs = plt.subplots(3,3, sharey=True, sharex=True, figsize=(9.3,10), dpi=300)
    
    axs[0,0].set_title('True models\n'+r'$\bf{d_{obs}}$')
    axs[0,1].set_title('Mean')
    axs[0,2].set_title('Residuals')
    axs[1,2].set_title('Variance')
    
    ax1 = axs[0,0].imshow(m_seis.detach().cpu(), cmap='seismic', vmin=-800, vmax=800)
    axs[0,1].imshow(mean_post_seis, cmap='seismic', vmin=-800, vmax=800)
    ax2 = axs[0,2].imshow(mean_post_seis-m_seis.detach().cpu().numpy(), cmap='seismic') 
    
    ax3 = axs[1,0].imshow(m_f,cmap='gray', vmin=0,vmax=1)
    axs[1,1].imshow(mean_post_f,cmap='gray', vmin=0,vmax=1)
    ax4 = axs[1,2].imshow(var_f,cmap='gray_r', vmax=np.round(var_f.max(),3))
    
    ax5 = axs[2,0].imshow(m_ip,cmap='jet', vmin=m_ip.min(), vmax=m_ip.max())
    axs[2,1].imshow(mean_post_ip,cmap='jet', vmin=m_ip.min(), vmax=m_ip.max())
    ax6 = axs[2,2].imshow(var_ip,cmap='Reds')
    
    fig.subplots_adjust(hspace= -0.2)
    
    for i in range(3):
        bbox = axs[i,0].get_position()
        bbox.x0-=0.045;bbox.x1-=0.045
        axs[i,0].set_position(bbox)
        
        bbox = axs[i,1].get_position()
        bbox.x0-=0.08;bbox.x1-=0.08
        axs[i,1].set_position(bbox)
                
        bbox = axs[i,2].get_position()
        bbox.x0-=0.015;bbox.x1-=0.015
        axs[i,2].set_position(bbox)
   
    bbox= axs[0,1].get_position()
    cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
    fig.colorbar(ax1, cax=cax)
    cax.set_ylabel('Ampl.',labelpad=-25, fontsize=15)
    cax.set_yticks([-800,0,800])
    
    bbox= axs[1,1].get_position()
    cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
    fig.colorbar(ax3, cax=cax)
    cax.set_ylabel('Facies', labelpad=5, fontsize=15)
    cax.set_yticks([0,1])

    bbox= axs[2,1].get_position()
    cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
    fig.colorbar(ax5, cax=cax)
    cax.set_ylabel(r'[m/s g/cm$^3$]',labelpad=-25, fontsize=15)
    cax.set_yticks([m_ip.min(),m_ip.max()])

    bbox= axs[0,2].get_position()
    cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
    fig.colorbar(ax2, cax=cax)
    
    bbox= axs[1,2].get_position()
    cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
    fig.colorbar(ax4, cax=cax)
    cax.set_yticks([0,np.round(var_f.max(),3)])
    
    bbox= axs[2,2].get_position()
    cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
    fig.colorbar(ax6, cax=cax)
    cax.set_yticks([0,var_ip.max()/2,var_ip.max()])

                  
    if setup.outdir[6] != '3':
        for i in range (3): 
            axs[2,i].xaxis.set_visible(False)
    else:
        axs[2,0].set_xlabel('CMP [m]')
        axs[2,0].set_ylabel('TWT [ms]')

    plt.savefig(workdir+'Inv_res_1.png')
    
    # %% Plot loss NT
    fig, axs = plt.subplots(1,1, figsize=(8,5))
    loss_p = axs.plot(np.arange(0, len(loss_vals)), loss_vals,
                        label='$-\mathcal{L}(\phi)$', color='#191970')
    axs.set_xlabel('Epoch', fontsize=20)
    axs.set_ylabel('Loss', color='#191970', fontsize=20)
    axs.set_xlim([0, len(loss_vals)])
    axs.set_xlim([0, int(step) if step!='last' else len(r_misfit)])
    axs.tick_params(axis='y', labelcolor='#191970')
    axs.set_yscale('log')
    
    ax5 = axs.twinx()
    ax5.set_yscale('log')
    
    
    ax5.set_ylabel(r'RMSE [Amplitude]', color='#c32148', fontsize=18)
    for kk in range(rmse_d.shape[-1]): misf_p = ax5.plot(np.arange(0, len(rmse_d[1:])), 
                                         rmse_d[1:], label='Data misfit', color='#c32148')
    
    sigma_p = ax5.plot(np.arange(0, len(rmse_d[1:])), 
                       np.ones(len(rmse_d[1:]))*mean_sigma.numpy(), 
                       '--', color='#c32148', linewidth='2', 
                       label='$\sigma={:.3f}~$'.format(mean_sigma.detach().cpu().numpy()))
    
    ax5.tick_params(axis='y', labelcolor='#c32148')

    lns = loss_p + misf_p +  sigma_p
    
    
    # lns = loss_p + misf_p +  sigma_p
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels)
    plt.tight_layout()
    plt.savefig(workdir+'Inv_res_2.png',  )

    # %% Plot grids MCMC
    if plot_MCMC:
        MCMC_samples = np.empty((0,Sequences[np.int32(index*0.5):index:8,0,0].size*Sequences[0,0,:].size))
        
        for i in range(MCMCPar.n):
            MCMC_samples = np.append(MCMC_samples, Sequences[np.int32(index*0.5):index:8,i,:].reshape((1,-1)), axis=0)
  
        mean_f= torch.empty(100,Sequences[0,0,:].size,80,100)
        mean_ip= torch.empty(100,Sequences[0,0,:].size,80,100)
        jj=0
        for j in range(index-100,index):
            samp= Sequences[j,0:MCMCPar.n].T
            post_MCMC_f,post_MCMC_ip= setup.gen(torch.tensor(samp, dtype=torch.float32))
            if setup.DGM=='GAN': post_MCMC_f= (post_MCMC_f + 1) * 0.5
            mean_f[jj]= post_MCMC_f.squeeze()
            mean_ip[jj]= post_MCMC_ip.squeeze()
            jj+=1
        mean_ip = (setup.ipmax-setup.ipmin)*mean_ip.clone()+setup.ipmin
        
        
        log_file.write(f'mean RMSEF McMC: {torch.sqrt(torch.sum((mean_f.mean(dim=1)-m_f).reshape(-1,8000)**2, dim=1)/8000).mean()}\n')
        log_file.write(f'std RMSEF McMC: {torch.sqrt(torch.sum((mean_f.mean(dim=1)-m_f).reshape(-1,8000)**2, dim=1)/8000).std()}\n')
        
        log_file.write(f'mean RMSEIP McMC: {torch.sqrt(torch.sum((mean_ip.mean(dim=1)-m_ip).reshape(-1,8000)**2, dim=1)/8000).mean()}\n')
        log_file.write(f'std RMSEIP McMC: {torch.sqrt(torch.sum((mean_ip.mean(dim=1)-m_ip).reshape(-1,8000)**2, dim=1)/8000).std()}\n')
        
        
        
        mean_seis = fm.fullstack_solver_simple(mean_ip.mean(dim=1)[:,None,:], setup, 'noise').detach().squeeze()
        
        log_file.write(f'mean RMSEd McMC: {torch.sqrt(torch.sum((mean_seis-m_seis).reshape(-1,8000)**2, dim=1)/8000).mean()}\n')
        log_file.write(f'std RMSEd McMC: {torch.sqrt(torch.sum((mean_seis-m_seis).reshape(-1,8000)**2, dim=1)/8000).std()}\n')
  
        
        var_f= torch.var(mean_f,axis=0).mean(axis=0)   
        mean_f= torch.mean(mean_f,axis=0).mean(axis=0)   
        
        var_ip= torch.var(mean_ip, axis=0).mean(axis=0)   
        mean_ip= torch.mean(mean_ip,axis=0).mean(axis=0)    
        mean_seis= torch.mean(mean_seis, axis=0)
        
        if DoThreshold:
            threshold = 0.5
            mean_f[mean_f < threshold] = 0
            mean_f[mean_f >= threshold] = 1
          
        fig, axs = plt.subplots(3,3, sharey=True, sharex=True, figsize=(11,11))
        
        axs[0,0].set_title('True models\n'+r'$\bf{d_{obs}}$')
        axs[0,1].set_title('Mean')
        axs[0,2].set_title('Residuals')
        axs[1,2].set_title('Variance')
        
        ax1 = axs[0,0].imshow(m_seis.detach().cpu(), cmap='seismic', vmin=-800, vmax=800)
        axs[0,1].imshow(mean_seis, cmap='seismic', vmin=-800, vmax=800)
        ax2 = axs[0,2].imshow(mean_seis-m_seis.detach().cpu().numpy(), cmap='seismic') 
        
        ax3 = axs[1,0].imshow(m_f,cmap='gray', vmin=0,vmax=1)
        axs[1,1].imshow(mean_f,cmap='gray', vmin=0,vmax=1)
        ax4 = axs[1,2].imshow(var_f,cmap='gray_r', vmax=np.round(var_f.max(),3))
        
        ax5 = axs[2,0].imshow(m_ip,cmap='jet', vmin=m_ip.min(), vmax=m_ip.max())
        axs[2,1].imshow(mean_ip,cmap='jet', vmin=m_ip.min(), vmax=m_ip.max())
        ax6 = axs[2,2].imshow(var_ip,cmap='Reds')
        
        fig.subplots_adjust(hspace= -0.2)
        
        for i in range(3):
            bbox = axs[i,0].get_position()
            bbox.x0-=0.04;bbox.x1-=0.04
            axs[i,0].set_position(bbox)
            
            bbox = axs[i,1].get_position()
            bbox.x0-=0.07;bbox.x1-=0.07
            axs[i,1].set_position(bbox)
                    
            bbox = axs[i,2].get_position()
            bbox.x0+=0.01;bbox.x1+=0.01
            axs[i,2].set_position(bbox)
       
        bbox= axs[0,1].get_position()
        cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
        fig.colorbar(ax1, cax=cax)
        cax.set_ylabel('Amplitude')
        cax.set_yticks([-800,0,800])
        
        bbox= axs[1,1].get_position()
        cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
        fig.colorbar(ax3, cax=cax)
        cax.set_ylabel('Facies', labelpad=30)
        cax.set_yticks([0,1])

        bbox= axs[2,1].get_position()
        cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
        fig.colorbar(ax5, cax=cax)
        cax.set_ylabel(r'I$_P$ '+r'[m/s g/cm$^3$]')
        cax.set_yticks([m_ip.min(),m_ip.max()])



        bbox= axs[0,2].get_position()
        cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
        fig.colorbar(ax2, cax=cax)
        # cax.set_ylabel('Amplitude', labelpad=-20)
        #
        
        
        bbox= axs[1,2].get_position()
        cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
        fig.colorbar(ax4, cax=cax)
        cax.set_yticks([0,np.round(var_f.max(),3)])
        


        bbox= axs[2,2].get_position()
        cax= fig.add_axes([bbox.x1+0.01,bbox.y0, 0.02, (bbox.y1-bbox.y0)])
        fig.colorbar(ax6, cax=cax)
        # cax.set_ylabel(r'[m/s g/cm$^3$]', labelpad=-25)
        cax.set_yticks([0,var_ip.max()/2,var_ip.max()])
   
            
        plt.savefig(workdir+'Inv_res_MCMC1.png',dpi=400,  )    
        # %% Plot loss MCMC
        fig, axs = plt.subplots(1,1, figsize=(8,5))
        
        axs.set_xlabel('Epoch', fontsize=20)
        # axs.set_ylabel('Loss', color='#191970', fontsize=20)
        axs.tick_params(axis='y', labelcolor='#c32148')
        axs.set_yscale('log')

        axs.set_ylabel(r'RMSE [Amplitude]', color='#c32148', fontsize=18)

        sigma_p = axs.plot(np.arange(0, MCMC_misfit.shape[0]*MCMCPar.thin), 
                           np.ones(MCMC_misfit.shape[0]*MCMCPar.thin)*mean_sigma.detach().cpu().numpy(), 
                               '--', color='#c32148', linewidth='2', 
                               label='$\sigma={:.3f}~$'.format(mean_sigma.detach().cpu().numpy()))
        
        misf_MCMC=axs.plot(np.linspace(0,MCMC_misfit.shape[0]*MCMCPar.thin, MCMC_misfit.shape[0]), 
                           MCMC_misfit, color='#c32148',label = 'MCMC')

        axs.set_xlim([0,MCMC_misfit.shape[0]*MCMCPar.thin])
        
        lns = loss_p + [misf_MCMC[0]] + sigma_p
        labels = [l.get_label() for l in lns]
        plt.legend(lns, labels)
        plt.tight_layout()

            
        plt.savefig(workdir+'Inv_res_MCMC2.png',dpi=400,  ) 
        
        
        
        
        # plot all losses
        fig, axs = plt.subplots(1,1, figsize=(8,5))
        axs.set_xlabel('Epoch (NT)', fontsize=20)
        axs.set_ylabel('Loss', color='#191970', fontsize=20)
        axs.set_xlim([0, int(step) if step!='last' else len(r_misfit)])
        axs.tick_params(axis='y', labelcolor='#191970')
        axs.set_yscale('log')
        
        
        loss_p = axs.plot(np.arange(0, len(loss_vals)), loss_vals,
                            label='$-\mathcal{L}(\phi)$ (NT)', color='#191970')
               
            #RMSE 
        ax1 = axs.twinx()
        ax1.set_yscale('log')
        ax1.set_ylabel(r'RMSE [Amplitude]', color='#c32148', fontsize=18)
        
            #NT
        for kk in range(rmse_d.shape[-1]): misf_p = ax1.plot(np.arange(0, len(rmse_d[1:])), 
                                             rmse_d[1:], label='Data misfit (NT)', color='#c32148')
        
        sigma_p = ax1.plot(np.arange(0, len(rmse_d[1:])), 
                           np.ones(len(rmse_d[1:]))*mean_sigma.numpy(), 
                           '--', color='#c32148', linewidth='2', 
                           label='$\sigma={:.3f}~$'.format(mean_sigma.detach().cpu().numpy()))
        
        ax1.tick_params(axis='y', labelcolor='#c32148')
        
        ax2 = ax1.twiny()
        ax2.set_xlabel('Iteration (McMC)', fontsize=20)
        ax2.tick_params(axis='y', labelcolor='#c32148')
        misf_MCMC=ax2.plot(np.linspace(0,MCMC_misfit.shape[0]*MCMCPar.thin, MCMC_misfit.shape[0]), 
                           MCMC_misfit, color='#f77205',label = 'Data misfit (McMC)')
        ax2.set_xlim([0,MCMC_misfit.shape[0]*MCMCPar.thin])
        
        lns = loss_p + misf_p + [misf_MCMC[0]] + sigma_p
        labels = [l.get_label() for l in lns]
        plt.legend(lns, labels)
        plt.tight_layout()

        plt.savefig(workdir+'Inv_res_3.png',dpi=400,  ) 
        
        
# %%            
    import seaborn as sns
    import pandas as pd
    if plot_latent:
            
        if plot_MCMC:
            oo= [False,True]
        else: 
            oo= [False]
        
        for plot_MCMC in oo:
            fig, axs = plt.subplots(3,3, figsize=(11,11), sharey=False, constrained_layout=True)
            plt.subplots_adjust(hspace=0.5)
            i = 0
            for row in range(3):
                for col in range(3):
                    #ax = fig.add_subplot(gs[row, col], sharex=None, sharey=None)
                    if i < num_pars:
                        axs[row, col].set(xlabel='', ylabel='PDF')
                        #axs[row, col].label_outer()
                        
                        if col==0: axs[row,col].yaxis.set_visible(True)
                        else: axs[row,col].yaxis.set_visible(False)
                        
                        if plot_MCMC:
                            sns.kdeplot(x=MCMC_samples[i,-500:],
                                        ax=axs[row, col], label='MCMC', fill=True, color='#089421')
                            
                        xmin, xmax = axs[row, col].get_xlim()
                        sns.kdeplot(x=posterior_samples[:, i].detach().cpu().numpy(),
                                    ax=axs[row, col], label='NT', fill=True, color='#00008b')
                        
                        xmin2, xmax2 = axs[row, col].get_xlim()
                        sns.kdeplot(x=base_samples[:, i],
                                    ax=axs[row, col], label='Prior', fill=True, color='#fb9902')
                        
                        if 'zs' in setup.__dict__:
                            axs[row, col].axvline(setup.zs.reshape(-1).cpu()[i], color='#ff0800',
                                    linestyle='dotted', linewidth=2.5, label='True value')
                            # xmax3= setup.zs.reshape(-1).max()
                            # xmin3=setup.zs.reshape(-1).min()
                            # xx= setup.zs.reshape(-1).cpu()[i]   
                            #xmax_t = setup.zs.reshape(-1).cpu()[i]+0.3
                            #xmin_t = setup.zs.reshape(-1).cpu()[i]-0.3                   
                        
                        xmax_t= max(xmax,xmax2)
                        xmin_t= min(xmin,xmin2)                        
    
                        # axs[row, col].xaxis.set_tick_params(labelbottom=True, labelsize=8)
                        # axs[row, col].tick_params(axis='both', which='minor', labelsize=8)
                        axs[row, col].yaxis.set_tick_params(labelleft=True)
                        
                        axs[row, col].set_xlim([xmin_t, xmax_t])
                        if i== num_pars-1: 
                            bbox= axs[row, col].get_position()
                            # axs[row, col].legend(bbox_to_anchor=[bbox.x1+3,bbox.y0+1])                
                    else: 
                        axs[row, col].set_visible(False)
                    i += 1
                    plt.legend(loc=[0.7,0.7])
                    # 
            plt.savefig(workdir+f'Inferred_Zspace_{plot_MCMC}.png',  )
            plt.close('all')
            
        
                
        
    # graph = pyro.render_model(guide, model_args=(setup,), filename='model.pdf')
    if plot_latent_step:
        Z = 0
        colors = plt.cm.rainbow(np.linspace(0.1, 0.9, 8))
        plt.figure(5, figsize=(5, 5), constrained_layout=True)
        sns.kdeplot(x=base_samples[:, Z], fill=True, label='Prior',color=colors[0])
        plt.xlabel('Z$_{%d}$' %(Z+1), fontsize=18)
        plt.ylabel('PDF', fontsize=18)
        i = 1
        for step in [50, 100, 150, 200,250, step]:
        # for step in [10,50,100,150,200,250,step]:
    
            pyro.clear_param_store()
            guide.load_state_dict(torch.load(
                workdir+'saved_NT_params_step_'+str(step)+'.pt'))
            pyro.set_rng_seed(1)
            posterior = guide.get_posterior()  # give samples from the posterior
            posterior_samples = posterior.rsample(
                sample_shape=torch.Size([num_samples]))
            sns.kdeplot(x=posterior_samples[:, Z].detach().numpy(
            ), fill=True, label=str(step) if step != 'last' else str(len(loss_vals)), color=colors[i])
            i+=1
        if 'zs' in setup.__dict__:
            plt.axvline(setup.zs.reshape(-1)[Z], color='r',
                        linestyle='dotted', linewidth=2.5, label='True z')
        # plt.xlim([-3, 3])
        plt.xlim([-4,4])
        plt.legend(loc='upper left')
        
        plt.savefig(workdir+'Inferred_Z_steps.png',  )
    plt.close('all')
    
    
    # %%
    if plot_MCMC:
        MCMC_samples = np.empty((0,Sequences[np.int32(index*0.5):index:8,0,0].size*Sequences[0,0,:].size))
        for i in range(MCMCPar.n):
            MCMC_samples = np.append(MCMC_samples, Sequences[np.int32(index*0.5):index:8,i,:].reshape((1,-1)), axis=0)
    
        def kl_divergence(p, q):
            p = p/np.sum(p)
            q = q/np.sum(q)
            # return sum(p * np.log(p/q))# for i in range(len(p)))
        # def kl_divergence(p, q):
            return np.sum(np.where(p > 1e-14, p * np.log(p / q), 0))
        
        from scipy import stats
        posterior_samples = posterior_samples.T.detach().cpu().numpy()
        prior = np.random.multivariate_normal(np.zeros(setup.total_event_size),np.eye(setup.total_event_size),num_samples).T
        
        x_data = np.arange(-5, 5, 0.001)
          
        ## y-axis as the gaussian
        # y_data = stats.norm.pdf(x_data, 0, 1)
        
        logS = np.zeros((3,setup.total_event_size))
        KL = np.zeros((3,setup.total_event_size))
        
        zz= setup.zs.detach().numpy().reshape(-1)
        for i in range(setup.total_event_size):
            
            kde_NT = stats.gaussian_kde(posterior_samples[i])
            kde_MCMC = stats.gaussian_kde(MCMC_samples[i,-num_samples:])
            kde_prior = stats.gaussian_kde(prior[i])
            
            logS[0,i] = -kde_NT.logpdf(zz[i])
            logS[1,i] = -kde_MCMC.logpdf(zz[i])
            logS[2,i]  = -kde_prior.logpdf(zz[i])
        
            KL[0,i] = kl_divergence(kde_NT.pdf(x_data),kde_MCMC.pdf(x_data))
            KL[1,i] =  kl_divergence(kde_NT.pdf(x_data),kde_prior.pdf(x_data))
            KL[2,i] =  kl_divergence(kde_MCMC.pdf(x_data),kde_prior.pdf(x_data))
        

        
        print(f'logS for NT: {np.mean(logS,axis=1)[0]}')
        print(f'logS for MCMC: {np.mean(logS,axis=1)[1]}')
        print(f'logS for prior: {np.mean(logS,axis=1)[2]}')
        log_file.write(f'\n\nlogS for NT: {np.mean(logS,axis=1)[0]}\n')
        log_file.write(f'logS for MCMC: {np.mean(logS,axis=1)[1]}\n')
        log_file.write(f'logS for prior: {np.mean(logS,axis=1)[2]}\n')
        
        
        # print('Neural transport logS = {:.2f}'.format(logS_NT[0])+'\n\nMCMC logS = {:.2f}'.format(logS_MCMC[0])+'\n\nPrior logS = {:.2f}'.format(logS_prior[0]))
              
        print('\n\nKLD Q(NT)||P(MCMC) = {:.2f}'.format(np.mean(KL,axis=1)[0]))
        print('KLD Q(NT)||P(prior) = {:.2f}'.format(np.mean(KL,axis=1)[1]))
        print('KLD Q(MCMC)||P(prior) = {:.2f}'.format(np.mean(KL,axis=1)[2]))
          
        log_file.write('\nKLD Q(NT)||P(MCMC) = {:.2f}'.format(np.mean(KL,axis=1)[0]))
        log_file.write('\nKLD Q(NT)||P(prior) = {:.2f}'.format(np.mean(KL,axis=1)[1]))
        log_file.write('\nKLD Q(MCMC)||P(prior) = {:.2f}'.format(np.mean(KL,axis=1)[2]))
