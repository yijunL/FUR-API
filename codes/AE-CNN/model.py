import torch
import torch.nn as nn
import os




class CNNAEModel(nn.Module):
    def __init__(self, encoder):
        super(CNNAEModel, self).__init__()

        self.cnn_model = encoder
        

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

        cls_vector = self.cnn_model(input_ids)
        # print(cls_vector.shape)

        encoded = self.encoder(cls_vector)

        decoded = self.decoder(encoded)

        return decoded, cls_vector
