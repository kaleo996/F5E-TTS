# import sys
# sys.path.append('..')
# import monotonic_align

import math
import torch
import torch.nn as nn

from .utils import SpkEncoder
from .utils import EmbeddingTable, Linear, ConvPositionEmbed, Mlp, MultiHeadAttention, TransformerBlock, FinalLayer
from .utils import Conv1dGLU


# 混元 GPT-4o-TTS 的 duration predictor
class DurationNarPredictor(nn.Module):
    def __init__(self, config):
        super(DurationNarPredictor, self).__init__()
        self.vocab_size = config.text_vocab_size
        self.model_dim = config.duration.hidden_size
        
        self.text_embedding = EmbeddingTable(
                num_embeddings = self.vocab_size,
                embedding_dim = self.model_dim)
             
        self.text_proj = Linear(self.model_dim, self.model_dim)

        # Convolutional positional encoder
        self.duration_pre = ConvPositionEmbed(
            hidden_size=self.model_dim,
            kernel_size=31,
            groups=16,
            causal=False,
        )
        self.spk_encoder = SpkEncoder()
        self.spk_proj = Linear(self.model_dim, self.model_dim) # spk embedding 输出维度已手工设置成 256（model_dim）
        # Build transformers
        self.duration_mid = nn.ModuleList()
        for _ in range(config.duration.num_layers):
            attn_block = MultiHeadAttention(**config.duration)
            ffn_block = Mlp(act_layer=lambda: nn.GELU(approximate="tanh"), **config.duration)
            self.duration_mid.append(TransformerBlock(attn_block, ffn_block, **config.duration))

        self.output_dim = 1
        self.duration_post = FinalLayer(self.model_dim, self.output_dim)
        self.duration_projm =  nn.Conv1d(self.model_dim, 100, 1)


    def preprocess(self, y, y_lengths, y_max_length):
        n_sqz = 2
        if y_max_length is not None:
            y_max_length = (y_max_length // n_sqz) * n_sqz
            y = y[:,:,:y_max_length]
            y_lengths = (y_lengths // n_sqz) * n_sqz
        return y, y_lengths, y_max_length

    # 训练用
    def forward(self, text, mels, text_lens=None, mel_lengths=None, frac_lengths_mask=(0.7, 1.0)):
        pass

    # 推理用
    # 输入：tokenize 成整数 id 的文本、prompt 音频 + 噪声的梅尔谱、盖掉梅尔谱中噪声和 padding 用的 mask
    # 输出：一个表示每条音频长度的数组
    @torch.no_grad()
    def inference(self, text, mels=None, mask=None):
        if mels is not None:
            if mask is not None:
                spk_emb = self.spk_encoder(mels, mask)
            else:
                spk_emb = self.spk_encoder(mels)
        else:
            spk_emb = None
        spk_emb = self.spk_proj(spk_emb)
        text = self.text_proj(self.text_embedding(text))
        ut = self.duration_pre(text) + text
        for block in self.duration_mid:
            ut = block(ut, condition=spk_emb)
        logw = self.duration_post(ut, spk_emb)
        w = torch.exp(logw.squeeze(-1))
        w_ceil = torch.ceil(w).int() # 保留每一个音素的长度，不要计算音频的总长度
        return w_ceil
    

# StableTTS 的 speaker encoder
class MelStyleEncoder(nn.Module):
    def __init__(
        self,
        n_mel_channels=100,
        style_hidden=128,
        style_vector_dim=256, # TODO 需要用一个统一的参数在 yaml 里控制
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
        # x = x.transpose(1, 2)

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


# StableTTS 的 duration predictor
class StableTTSDurationPredictor(nn.Module):
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
        # import ipdb; ipdb.set_trace()
        x = x + self.cond(g.unsqueeze(2).detach())
        # try:
        x = self.conv1(x * x_mask)
        # except:
        #     print("x shape is: ", x.shape)
        #     print("x_mask shape is: ", x_mask.shape)
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
