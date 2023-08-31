import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel
import os


# 创建自定义模型
class BertAEModel(nn.Module):
    def __init__(self):
        super(BertAEModel, self).__init__()
        # 加载预训练的BERT模型
        bert_config = BertConfig.from_pretrained("../bert-base-uncased", num_labels=2)
        self.bert_model = BertModel.from_pretrained(
            os.path.join("../bert-base-uncased", "pytorch_model.bin"),
            config=bert_config
        )
        # 自编码机的编码器和解码器部分
        
        self.encoder1 = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(96, 192),
            nn.ReLU(), 
            nn.Linear(192, 384),
            nn.ReLU(),
            nn.Linear(384, 768),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
        )
        # 分类任务的输出层
        # self.classifier = nn.Linear(768, 2)  # 2 classes: normal and anomaly

    def forward(self, input_ids, attention_mask):
        # 使用BERT编码文本字段
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        cls_vector = bert_output.pooler_output  # 提取CLS向量
        # print(cls_vector.shape)
        # 编码器部分
        z1 = self.encoder1(cls_vector)
        # 解码器部分
        x1 = self.decoder(z1)
        
        z2 = self.encoder2(x1)
        
        z3 = self.discriminator(x1)
        
        z4 = self.discriminator(cls_vector)
        # 分类任务
        # classification_output = self.classifier(decoded)  # 使用[CLS]的输出
        return cls_vector, x1, z1, z2, z3, z4
