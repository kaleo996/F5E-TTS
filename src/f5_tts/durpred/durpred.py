import torch
import torch.nn as nn

from .utils import Conv1dGLU


# speaker encoder of StableTTS
class MelStyleEncoder(nn.Module):
    def __init__(
        self,
        n_mel_channels=100,
        style_hidden=128,
        style_vector_dim=256,
        style_kernel_size=5,
        style_head=2,
        dropout=0.1,
    ):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        self.spectral = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = nn.MultiheadAttention(
            self.hidden_dim,
            self.n_head,
            self.dropout,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            return torch.sum(x * ~mask.unsqueeze(-1), dim=1) / (~mask).sum(dim=1).unsqueeze(1)

    def forward(self, x, x_mask=None):
        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        if x_mask is not None:
            x_mask = ~x_mask.squeeze(1).to(torch.bool)   
        x, _ = self.slf_attn(x, x, x, key_padding_mask=x_mask, need_weights=False)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=x_mask)

        return w


# duration predictor of StableTTS
class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, style_vector_dim=0):
        super().__init__()

        self.drop = nn.Dropout(p_dropout)
        self.conv1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(filter_channels)
        self.conv2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        self.cond = nn.Conv1d(style_vector_dim, in_channels, 1)

    def forward(self, x, x_mask, g):
        x = x.detach().transpose(1, 2)
        x_mask = x_mask.transpose(1, 2)
        x = x + self.cond(g.unsqueeze(2).detach())
        x = self.conv1(x * x_mask)
        x = torch.relu(x)
        x = self.norm1(x.transpose(1,2)).transpose(1,2)
        x = self.drop(x)
        x = self.conv2(x * x_mask)
        x = torch.relu(x)
        x = self.norm2(x.transpose(1,2)).transpose(1,2)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        x =  x * x_mask
        return x
