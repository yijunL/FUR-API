import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel
import os
import torch.nn.functional as F

# 创建自定义模型
class BertVAEModel(nn.Module):
    def __init__(self):
        super(BertVAEModel, self).__init__()
        # 加载预训练的BERT模型
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

        self.fc_mu = nn.Linear(128, self.latent_dim)  # 编码器的均值输出
        self.fc_logvar = nn.Linear(128, self.latent_dim)  # 编码器的对数方差输出

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布中采样噪声
        z = mu + eps * std  # 重参数化
        return z

    def forward(self, input_ids, attention_mask):
        # 使用BERT编码文本字段
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        cls_vector = bert_output.pooler_output  # 提取CLS向量
        # print(cls_vector.shape)

        # 编码器部分
        x = self.encoder(cls_vector)

        # 重参数化并采样
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # 解码器部分
        reconstruction = self.decoder(z)

        return cls_vector, reconstruction, mu, logvar

    def compute_loss(self, x, recon_x, mu, logvar):
        # 计算重构损失  均方误差损失
        # criterion = nn.MSELoss(reduction='sum')  # 创建损失函数实例
        # recon_loss = criterion(recon_x, x)  # 计算重构损失
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        # 计算 KL 散度项，用于正则化潜在空间
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         print("recon_loss:",recon_loss, "   kl_loss:",kl_loss)
        return recon_loss + 0.5*kl_loss
    
    def compute_rec_loss(self, x, recon_x, mu, logvar):
        # 计算重构损失  均方误差损失
        # criterion = nn.MSELoss(reduction='sum')  # 创建损失函数实例
        # recon_loss = criterion(recon_x, x)  # 计算重构损失
#         recon_loss = F.mse_loss(recon_x, x, reduction='mean')

#         return recon_loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        # 计算 KL 散度项，用于正则化潜在空间
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         print("recon_loss:",recon_loss, "   kl_loss:",kl_loss)
        return recon_loss + 0.5*kl_loss