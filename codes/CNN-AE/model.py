import torch
import torch.nn as nn
import os



# 创建自定义模型
class CNNAEModel(nn.Module):
    def __init__(self, encoder):
        super(CNNAEModel, self).__init__()

        self.cnn_model = encoder
        
        # 自编码机的编码器和解码器部分
        self.encoder = nn.Sequential(
            nn.Linear(230, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 230),
        )

    def forward(self, input_ids):
        # 使用CNN编码文本字段
        cls_vector = self.cnn_model(input_ids)
        # print(cls_vector.shape)
        # 编码器部分
        encoded = self.encoder(cls_vector)
        # 解码器部分
        decoded = self.decoder(encoded)
        # 分类任务
        # classification_output = self.classifier(decoded)  # 使用[CLS]的输出
        return decoded, cls_vector
