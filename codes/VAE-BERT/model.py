import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel
import os
import torch.nn.functional as F

class BertVAEModel(nn.Module):
    def __init__(self):
        super(BertVAEModel, self).__init__()
        bert_config = BertConfig.from_pretrained("../bert-base-uncased", num_labels=2)
        self.latent_dim = 32
        self.bert_model = BertModel.from_pretrained(
            os.path.join("../bert-base-uncased", "pytorch_model.bin"),
            config=bert_config
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.fc_mu = nn.Linear(128, self.latent_dim) 
        self.fc_logvar = nn.Linear(128, self.latent_dim) 

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)  
        z = mu + eps * std 
        return z

    def forward(self, input_ids, attention_mask):

        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        cls_vector = bert_output.pooler_output
        # print(cls_vector.shape)


        x = self.encoder(cls_vector)


        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)


        reconstruction = self.decoder(z)

        return cls_vector, reconstruction, mu, logvar

    def compute_loss(self, x, recon_x, mu, logvar):

        # criterion = nn.MSELoss(reduction='sum')
        # recon_loss = criterion(recon_x, x)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         print("recon_loss:",recon_loss, "   kl_loss:",kl_loss)
        return recon_loss + 0.5*kl_loss
    
    def compute_rec_loss(self, x, recon_x, mu, logvar):

        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         print("recon_loss:",recon_loss, "   kl_loss:",kl_loss)
        return recon_loss + 0.5*kl_loss
