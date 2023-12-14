import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlattenL(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 512, 3, 4)
    
class UnFlatten2(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 512, 4, 4)

class VAE_2layersSmall(nn.Module):
    def __init__(self, image_channels=1, h_dim=12*256, z_dim=100, veps=1):
        super(VAE_2layersSmall, self).__init__()
        self.veps= veps
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(128), #nn.BatchNorm2d(32,momentum=0.01),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 512, kernel_size=3, stride=2, bias=False),
            nn.InstanceNorm2d(512), #nn.BatchNorm2d(64,momentum=0.01),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, kernel_size=(3, 4), stride=2, bias=False),
            nn.InstanceNorm2d(256), #
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=(3, 5), stride=2, bias=False),
            nn.InstanceNorm2d(512), #
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(4096*2,512*2),
            nn.LeakyReLU(0.2),
        )
        
        self.fc1 = nn.Linear(512*2, z_dim)
        self.fc2 = nn.Linear(512*2, z_dim)
        self.fc3 = nn.Linear(z_dim, 512*2)
        
        self.decoder = nn.Sequential(
            nn.Linear(512*2,4096*2),
            nn.LeakyReLU(0.2),
            UnFlatten2(),
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 5), stride=2, bias=False),
            nn.InstanceNorm2d(256), #,
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 512, kernel_size=(3, 4), stride=2, bias=False),
            nn.InstanceNorm2d(512), #,
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, bias=False),
            nn.InstanceNorm2d(128), #
            nn.LeakyReLU(0.2),
            )
        self.fnet = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2,stride=1, bias=False),
                                   nn.ReLU(),
                                  nn.ConvTranspose2d(64, 1, kernel_size=2,stride=2, bias=False),
                                  nn.Sigmoid())
        self.ipnet = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2,stride=1, bias=False),
                                    nn.ReLU(),
                                   nn.ConvTranspose2d(64, 1, kernel_size=2,stride=2, bias=False),
                                   nn.Sigmoid())

        
    def reparameterize(self, mu, logvar, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=device)
        z = mu + std * esp * self.veps
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        #fac= recon_x[:,0,None,:,:]
        #ip= recon_x[:,1,None,:,:]
        fac= self.fnet(recon_x)
        ip= self.ipnet(recon_x)
        
        return fac,ip, z, mu, logvar
    
    def forwardNT(self,z):
        recon_x = self.decode(z)
        fac= self.fnet(recon_x)
        ip= self.ipnet(recon_x)

        return fac, ip
		
class VAE_2layers(nn.Module):
    def __init__(self, image_channels=1, h_dim=12*256, z_dim=100, veps=1):
        super(VAE_2layers, self).__init__()
        self.veps= veps
        self.encoder = nn.Sequential(
            
            nn.Conv2d(2, 32, kernel_size=2, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32), #nn.BatchNorm2d(32,momentum=0.01),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64), #nn.BatchNorm2d(64,momentum=0.01),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128), #
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=(0,1), bias=False),
            nn.InstanceNorm2d(512), #
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, kernel_size=5, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256), #
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=5, stride=(2,3), padding=1, bias=False),
            nn.InstanceNorm2d(512), #
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(4096*2,512*2),
            nn.LeakyReLU(0.2),
        )
        
        self.fc1 = nn.Linear(512*2, z_dim)
        self.fc2 = nn.Linear(512*2, z_dim)
        self.fc3 = nn.Linear(z_dim, 512*2)
        
        self.decoder = nn.Sequential(
            nn.Linear(512*2,4096*2),
            nn.LeakyReLU(0.2),
            UnFlatten2(),
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
                                  nn.Sigmoid())
        
        self.ipnet = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=2,stride=1,padding=0,bias=False),
                                  nn.LeakyReLU(0.2),
                                  nn.ConvTranspose2d(32,1,kernel_size=2,stride=1,padding=1,bias=False),
                                  nn.Sigmoid())

        
    def reparameterize(self, mu, logvar, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=device)
        z = mu + std * esp * self.veps
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        #fac= recon_x[:,0,None,:,:]
        #ip= recon_x[:,1,None,:,:]
        fac= self.fnet(recon_x)
        ip= self.ipnet(recon_x)
        
        return fac,ip, z, mu, logvar
    
    def forwardNT(self,z):
        recon_x = self.decode(z)
        fac= self.fnet(recon_x)
        ip= self.ipnet(recon_x)

        return fac, ip



class VAE_Laloy(nn.Module):
    def __init__(self, image_channels=1, h_dim=12*256, z_dim=20, veps=1, cuda=True):
        super(VAE_Laloy, self).__init__()
        self.veps= veps
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(128), #nn.BatchNorm2d(32,momentum=0.01),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 512, kernel_size=3, stride=2, bias=False),
            nn.InstanceNorm2d(512), #nn.BatchNorm2d(64,momentum=0.01),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, kernel_size=(3, 4), stride=2, bias=False),
            nn.InstanceNorm2d(256), #
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=(3, 5), stride=2, bias=False),
            nn.InstanceNorm2d(512), #
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(4096*2,512*2),
            nn.LeakyReLU(0.2),
        )
        
        self.fc1 = nn.Linear(512*2, z_dim)
        self.fc2 = nn.Linear(512*2, z_dim)
        self.fc3 = nn.Linear(z_dim, 512*2)
        
        self.decoder = nn.Sequential(
            nn.Linear(512*2,4096*2),
            nn.LeakyReLU(0.2),
            UnFlatten2(),
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 5), stride=2, bias=False),
            nn.InstanceNorm2d(256), #,
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 512, kernel_size=(3, 4), stride=2, bias=False),
            nn.InstanceNorm2d(512), #,
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, bias=False),
            nn.InstanceNorm2d(128), #
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, image_channels, kernel_size=4, stride=2, bias=False),
            nn.Sigmoid(),
            )
        # if cuda:
        #     self.load_state_dict(torch.load(gpath))
        # else:
        #     self.load_state_dict(torch.load(gpath, map_location=lambda storage, loc: storage))
    
    def reparameterize(self, mu, logvar, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=device)
        z = mu + std * esp * self.veps
        return z    
    
    # def reparameterize(self, mu, logvar):
    #     # for generation no noise is added (mean autoencoder is used)
    #     #std = logvar.mul(0.5).exp_()
    #     #esp = torch.randn(mu.size())
    #     z = mu #+ std * esp
    #     return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)

        return recon_x[:,0,None,:,:],recon_x[:,1,None,:,:], z, mu, logvar
    
    def forwardNT(self, z):
        recon_x = self.decode(z)

        return recon_x[:,0,None,:,:],recon_x[:,1,None,:,:]

    