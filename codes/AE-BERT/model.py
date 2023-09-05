import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel
import os



class BertAEModel(nn.Module):
    def __init__(self):
        super(BertAEModel, self).__init__()

        bert_config = BertConfig.from_pretrained("../bert-base-uncased", num_labels=2)
        self.bert_model = BertModel.from_pretrained(
            os.path.join("../bert-base-uncased", "pytorch_model.bin"),
            config=bert_config
        )

        self.encoder = nn.Sequential(
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

        # self.classifier = nn.Linear(768, 2)  # 2 classes: normal and anomaly

    def forward(self, input_ids, attention_mask):

        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        cls_vector = bert_output.pooler_output  
        # print(cls_vector.shape)

        encoded = self.encoder(cls_vector)

        decoded = self.decoder(encoded)
        return decoded, cls_vector
