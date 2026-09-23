import torch
import torch.nn as nn


class FFN(nn.Module):

    def __init__(self, embed_dim=256, feedforward_channels=1024, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, feedforward_channels)
        self.activation = nn.ReLU()

        self.fc2 = nn.Linear(feedforward_channels, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
            FFN 前向传播 (只负责算出残差,残差做和在encoder层) 
            x: Tensor [B, N, C]
        """

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        x = self.dropout(x)

        return x
        