"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import math
import torch
import random
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
    get_pos_embed_indices,
    PPGInputTranspose,
    GumbelVectorQuantizer
)

from f5_tts.durpred import get_mask_from_lengths
import f5_tts.durpred.monotonic_align as monotonic_align

# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], batch, seq_len, drop_text=False):  # noqa: F722
        if text is None:
            text = torch.zeros((batch, seq_len)).int().to(self.text_embed.weight.device)
        else:
            text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
            text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
            batch, text_len = text.shape[0], text.shape[1]
            text = F.pad(text, (0, seq_len - text_len), value=0) # pad to the same length as mel
            if self.mask_padding:
                text_mask = text == 0

            if drop_text:  # cfg for text
                text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text


# PPG embedding


class PPGEmbedding(nn.Module):
    def __init__(self, ppg_dim, text_dim):
        super().__init__()
        self.ppg_dim = ppg_dim
        self.text_dim = text_dim
        self.ppg_proj = nn.Sequential(
            nn.Linear(ppg_dim, ppg_dim),
            PPGInputTranspose(),
            nn.Conv1d(ppg_dim, ppg_dim, kernel_size=5, padding="same"),
            nn.BatchNorm1d(ppg_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(ppg_dim, ppg_dim, kernel_size=5, padding="same"),
            nn.BatchNorm1d(ppg_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(ppg_dim, ppg_dim, kernel_size=5, padding="same"),
            nn.BatchNorm1d(ppg_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            PPGInputTranspose(),
            nn.Linear(ppg_dim, text_dim), # output ppg in the same dimension as text for alignment
        )
        
    def forward(self, ppg_embed: None | float["b n d"], seq_len,drop_ppg=False, batch=None):  # noqa: F722
        if ppg_embed is None:
            dtype = next(self.ppg_proj.parameters()).dtype
            ppg_embed = torch.zeros((batch, seq_len, self.ppg_dim), dtype=dtype).to(self.ppg_proj[0].weight.device)
        else:
            ppg_len = ppg_embed.shape[1]
            ppg_embed = F.pad(ppg_embed, (0,0,0, seq_len - ppg_len), value=0) # pad to the same length as mel
            if drop_ppg:  # cfg for ppg
                ppg_embed = torch.zeros_like(ppg_embed)
        ppg_embed = self.ppg_proj(ppg_embed)
        return ppg_embed


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, use_ppg):
        super().__init__()
        self.use_ppg = use_ppg
        if use_ppg:
            self.proj = nn.Linear(mel_dim * 2 + text_dim * 2, out_dim)
        else:
            self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], ppg_embed=None, drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
        if self.use_ppg:
            x = self.proj(torch.cat((x, cond, text_embed, ppg_embed), dim=-1))
        else:
            x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        long_skip_connection=False,
        checkpoint_activations=False,
        ppg_config=dict(use_ppg=False),
        cb_config=dict(use_codebook=False),
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        
        self.use_ppg = ppg_config["use_ppg"]
        if self.use_ppg:
            self.ppg_embed = PPGEmbedding(ppg_config["ppg_dim"], text_dim)
            self.use_cross_mask = ppg_config.get("use_cross_mask", False)
            if self.use_cross_mask:
                cross_mask_cfg = ppg_config["cross_mask_config"]
                self.cross_mask_prob = cross_mask_cfg["cross_mask_prob"]
            
        self.use_codebook = cb_config["use_codebook"]
        if self.use_codebook:
            self.quantizer = self.get_codebook(cb_config, text_dim)
            
            self.use_perplex_loss = cb_config.get("use_perplex_loss", False)
            if self.use_perplex_loss:
                perplex_loss_cfg = cb_config["perplex_loss_config"]
                self.perplex_loss_prob = perplex_loss_cfg["perplex_loss_prob"]
                self.perplex_loss_weight = perplex_loss_cfg["perplex_loss_weight"]
            
            self.use_align_loss = cb_config.get("use_align_loss", False)
            if self.use_align_loss:
                align_loss_cfg = cb_config["align_loss_config"]
                self.align_loss_weight = align_loss_cfg["align_loss_weight"]
                self.text_embed_to_mel_dim = nn.Linear(text_dim, mel_dim) # project text to mel_dim to perform MAS

        self.input_embed = InputEmbedding(mel_dim, text_dim, dim, self.use_ppg)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None
        
    def get_codebook(self, cb_config, text_dim):
        return GumbelVectorQuantizer(
                dim = text_dim, # dim of txt and ppg after embeddings
                num_vars = cb_config["num_vars"],
                temp = (cb_config["temp_start"], cb_config["temp_stop"], cb_config["temp_decay"]),
                groups = cb_config["groups"],
                combine_groups = cb_config["combine_groups"],
                vq_dim = text_dim, # dim of txt and ppg after vq
                time_first = True,
                weight_proj_depth = cb_config["weight_proj_depth"],
                weight_proj_factor = cb_config["weight_proj_factor"]
            )
    
    # use MAS to align text and mel
    def align_text_mel(self, text_embed, text_len, seq_len, mel, mel_mask):
        text_mask = get_mask_from_lengths(text_len, max_len=seq_len)
        text_mask = text_mask.to(text_embed.device)
        text_embed_in_mel_dim = self.text_embed_to_mel_dim(text_embed)
        
        text_embed_t = text_embed_in_mel_dim.transpose(1, 2) # [b, d, nt]
        mel_t = mel.transpose(1, 2) # [b, d, n]
        with torch.no_grad():
            s_p_sq_r = torch.ones_like(text_embed_t)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi)- torch.zeros_like(text_embed_t), [1], keepdim=True)
            neg_cent2 = torch.einsum("bdt, bds -> bts", -0.5 * (mel_t**2), s_p_sq_r)
            neg_cent3 = torch.einsum("bdt, bds -> bts", mel_t, (text_embed_t * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (text_embed_t**2) * s_p_sq_r, [1], keepdim=True)  
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            
            attn_mask = text_mask.unsqueeze(1) * mel_mask.unsqueeze(2)
            attn = (monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach())

        attn = attn.squeeze(1).transpose(1, 2)
        return attn
    
    def calc_align_loss_vectorized(self, attn, text_embed, text_len, ppg_embed, ppg_len, mel_mask):
        batch_size, max_text_len, max_mel_len = attn.shape
        device = attn.device

        text_embed = self.quantizer(text_embed)["x"]
        ppg_embed = self.quantizer(ppg_embed)["x"]

        # 构建 batch-wise 的梅尔谱时间步 [batch_size, max_mel_len]
        m_values = torch.arange(max_mel_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 获取当前 batch 中每条数据的 mel 长度 [batch_size]
        mel_lengths = mel_mask.long().sum(dim=1)  # [batch_size]

        # 构建归一化后的 PPG 时间索引 [batch_size, max_mel_len]
        p_values = (m_values.float() / (mel_lengths.unsqueeze(1) - 1).clamp(min=1e-6)) * (ppg_len.unsqueeze(1) - 1)

        # 分离整数部分与小数部分
        low = torch.floor(p_values).long()
        high = low + 1
        frac = p_values - low

        # clamp 超出范围的 high 和 low
        high = torch.clamp(high, max=ppg_len.unsqueeze(1) - 1)
        low = torch.where(high >= ppg_len.unsqueeze(1), ppg_len.unsqueeze(1) - 1, low)
        frac = torch.where(high >= ppg_len.unsqueeze(1), torch.zeros_like(frac), frac)

        # 批量 gather PPG embedding
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_mel_len)
        low_indices = low
        high_indices = high

        # 提取对应的 PPG 特征
        ppg_low = ppg_embed[batch_indices, low_indices]  # [batch_size, max_mel_len, ppg_dim]
        ppg_high = ppg_embed[batch_indices, high_indices]

        # 插值
        interp_m_batch = (1 - frac.unsqueeze(-1)) * ppg_low + frac.unsqueeze(-1) * ppg_high  # [batch_size, max_mel_len, ppg_dim]

        # mask 掉超出实际 mel length 的部分
        interp_m_batch = interp_m_batch * mel_mask.unsqueeze(-1).float()

        # 注意力加权平均
        sum_term = torch.einsum('btm,bmd->btd', attn.float(), interp_m_batch)
        count_term = attn.sum(dim=2, keepdim=True).float().clamp(min=1e-8)
        avg_ppg = sum_term / count_term

        # 创建文本掩码
        text_mask = torch.arange(max_text_len, device=device).unsqueeze(0) < text_len.unsqueeze(1).to(device)
        text_mask = text_mask.unsqueeze(-1).float()

        # 计算 L2 损失
        loss = F.mse_loss(text_embed * text_mask, avg_ppg * text_mask, reduction='sum')
        loss = loss / text_mask.sum()
        loss *= self.align_loss_weight

        return loss

    # quantize 10% of txt and ppg tokens
    # text_embed: [b, nt, d], text_len: [b], ppg_embed: [b, n, d], ppg_len: [b]
    def quantize_calc_perplex_loss(self, text_embed, ppg_embed, drop_text=False, drop_ppg=False):
        perplex_loss = 0

        if not drop_text:
            quantized_text = self.quantizer(text_embed) # quantized_text is a dict
            text_rand_idx = torch.randperm(text_embed.size(1))[:int(text_embed.size(1) * self.perplex_loss_prob)]
            quantized_text_weight = quantized_text["x"].new_zeros(text_embed.size(1))
            quantized_text_weight[text_rand_idx] = 1
            text_embed = quantized_text_weight.unsqueeze(1) * quantized_text["x"] + (1 - quantized_text_weight).unsqueeze(1) * text_embed
            perplex_loss += (quantized_text["num_vars"] - quantized_text["prob_perplexity"]) / quantized_text["num_vars"]

        if not drop_ppg:
            quantized_ppg = self.quantizer(ppg_embed)
            ppg_rand_idx = torch.randperm(ppg_embed.size(1))[:int(ppg_embed.size(1) * self.perplex_loss_prob)]
            quantized_ppg_weight = quantized_ppg["x"].new_zeros(ppg_embed.size(1))
            quantized_ppg_weight[ppg_rand_idx] = 1
            ppg_embed = quantized_ppg_weight.unsqueeze(1) * quantized_ppg["x"] + (1 - quantized_ppg_weight).unsqueeze(1) * ppg_embed
            perplex_loss += (quantized_ppg["num_vars"] - quantized_ppg["prob_perplexity"]) / quantized_ppg["num_vars"]

        perplex_loss *= self.perplex_loss_weight
        return text_embed, ppg_embed, perplex_loss

    def cross_mask(self, attn, text_embed, text_len, ppg_embed, ppg_len):
        device = text_embed.device
        batch, max_text_len, _ = text_embed.shape
        _, max_ppg_len, _ = ppg_embed.shape

        text_valid_mask = get_mask_from_lengths(text_len, max_len=max_text_len).to(device) # [batch, max_text_len]
        ppg_valid_mask = get_mask_from_lengths(ppg_len, max_len=max_ppg_len).to(device)

        # for each sample, randomly mask a continuous span of 30% ~ 70% valid text tokens
        text_len = text_len.to(device)
        mask_ratio = 0.3 + 0.4 * torch.rand(batch, device=device) # [batch]
        mask_len = (mask_ratio * text_len.float()).clamp(min=1).long()
        start_max = text_len - mask_len
        start_ratio = torch.rand(batch, device=device)
        start = (start_max * start_ratio).long()
        indices = torch.arange(max_text_len, device=device).expand(batch, -1)
        end = start + mask_len
        text_mask = (indices < start.unsqueeze(1)) | (indices >= end.unsqueeze(1))
        text_mask &= text_valid_mask # also mask the paddings after valid text tokens

        # find corresponding PPG mask using alignment
        ppg_to_text = attn.argmax(dim=1)  # [batch, max_ppg_len], which text token each PPG token corresponds to
        ppg_mask = text_mask.gather(1, ppg_to_text)  # [batch, max_ppg_len]
        ppg_mask = ~ppg_mask  # if the corresponding text token is reserved, then the PPG token is masked
        ppg_mask &= ppg_valid_mask

        masked_text_embed = text_embed.masked_fill(~text_mask.unsqueeze(-1), 0)
        masked_ppg_embed = ppg_embed.masked_fill(~ppg_mask.unsqueeze(-1), 0)

        return masked_text_embed, masked_ppg_embed
    
    def sample(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        ppg,
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        drop_ppg,
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        
        # cache when inferring
        if drop_text:
            if self.text_uncond is None:
                self.text_uncond = self.text_embed(text, batch, seq_len, drop_text=True)
            text_embed = self.text_uncond
        else:
            if self.text_cond is None:
                self.text_cond = self.text_embed(text, batch, seq_len, drop_text=False)
            text_embed = self.text_cond


        if self.use_ppg:
            ppg_embed = self.ppg_embed(ppg, seq_len=seq_len, drop_ppg=drop_ppg, batch=batch)
        else:
            ppg_embed = None

        x = self.input_embed(x, cond, text_embed, ppg_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        pred = self.proj_out(x)

        return pred

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        ppg,
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        drop_ppg,
        mask: bool["b n"] | None = None,  # noqa: F722
        text_len = None,  # text length
        ppg_len = None,
        mel = None, # target mel used to train duration predictor
        mel_mask: bool["b n"] | None = None, # mask of target mel # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        
        text_embed = self.text_embed(text, batch, seq_len, drop_text=drop_text)

        if self.use_ppg:
            ppg_embed = self.ppg_embed(ppg, seq_len=seq_len, drop_ppg=drop_ppg, batch=batch)
        else:
            ppg_embed = None

        extra_loss = 0
        use_both_modal = not drop_text and not drop_ppg

        if self.use_codebook:
            # alignment loss: find corresponding txt-ppg token pairs, measure the distance between two modalities
            if self.use_align_loss and use_both_modal:
                attn = self.align_text_mel(text_embed, text_len, seq_len, mel, mel_mask)
                align_loss = self.calc_align_loss_vectorized(attn, text_embed, text_len, ppg_embed, ppg_len, mel_mask)
                # check if align_loss is NaN
                if torch.isnan(align_loss).any():
                    print("align_loss is NaN")
                    align_loss = 0
                extra_loss += align_loss

            # perplexity loss: encourage codebook to use same group of quantized vectors for both txt and ppg modalities
            if self.use_perplex_loss:
                text_embed, ppg_embed, perplex_loss = self.quantize_calc_perplex_loss(text_embed, ppg_embed, drop_text=drop_text, drop_ppg=drop_ppg)
                extra_loss += perplex_loss
        
        if self.use_cross_mask and use_both_modal:
            attn = self.align_text_ppg(text_embed, text_len, ppg_embed, ppg_len)
            if random.random() < self.cross_mask_prob:
                text_embed, ppg_embed = self.cross_mask(attn, text_embed, text_len, ppg_embed, ppg_len)
        
        x = self.input_embed(x, cond, text_embed, ppg_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        if extra_loss > 0:
            return output, extra_loss
        else:
            return output
