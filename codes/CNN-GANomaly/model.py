import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel
import os



# 创建自定义模型
class CNNAEModel(nn.Module):
    def __init__(self, encoder):
        super(CNNAEModel, self).__init__()

        self.cnn_model = encoder
        
        # 自编码机的编码器和解码器部分
        self.encoder1 = nn.Sequential(
            nn.Linear(230, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 230),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(230, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(230, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, input_ids):
        # 使用CNN编码文本字段
        cls_vector = self.cnn_model(input_ids)

        z1 = self.encoder1(cls_vector)

        x1 = self.decoder(z1)
        
        z2 = self.encoder2(x1)
        
        z3 = self.discriminator(x1)
        
        z4 = self.discriminator(cls_vector)

        return cls_vector, x1, z1, z2, z3, z4
