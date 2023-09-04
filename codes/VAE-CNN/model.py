import torch
import torch.nn as nn
import os
import torch.nn.functional as F


class CNNAEModel(nn.Module):
    def __init__(self, encoder):
        super(CNNAEModel, self).__init__()

        self.cnn_model = encoder
        self.latent_dim = 32
        
        self.encoder = nn.Sequential(
            nn.Linear(230, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.fc_mu = nn.Linear(64, self.latent_dim)  
        self.fc_logvar = nn.Linear(64, self.latent_dim)  

        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 230),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)  
        z = mu + eps * std  
        return z

        
    def forward(self, input_ids):

        cls_vector = self.cnn_model(input_ids)
        # print(cls_vector.shape)

        x = self.encoder(cls_vector)


        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)


        reconstruction = self.decoder(z)

        return cls_vector, reconstruction, mu, logvar

    def compute_loss(self, x, recon_x, mu, logvar):

        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + 0.5*kl_loss
    
    def compute_rec_loss(self, x, recon_x, mu, logvar):

        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        return recon_loss
