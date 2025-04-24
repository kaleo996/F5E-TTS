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
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNorm_Final,
    PPGInputTranspose,
    precompute_freqs_cis,
    get_pos_embed_indices,
    GumbelVectorQuantizer
)

from f5_tts.durpred import MelStyleEncoder, DurationPredictor, duration_loss
from f5_tts.durpred import sequence_mask, generate_path, get_mask_from_lengths
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
            ppg_embed = torch.zeros((batch, seq_len, self.ppg_dim)).to(self.ppg_proj[0].weight.device)
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
        ppg_config=dict(),
        cb_config=dict(),
        durpred_config=dict(),
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
            
        self.use_codebook = cb_config["use_codebook"]
        if self.use_codebook:
            self.codebook_prob = cb_config["codebook_prob"]
            self.codebook_loss_weight = cb_config["codebook_loss_weight"]
            self.quantizer = self.get_codebook(cb_config, text_dim)
        
        self.use_durpred = durpred_config["use_durpred"]
        if self.use_durpred:
            self.style_vector_dim = durpred_config["style_vector_dim"]
            self.durpred_text_embed = nn.Linear(text_dim, mel_dim) # project text to mel_dim, so that monotonic alignment search can be performed
            self.spk_encoder, self.durpred = self.get_durpred(durpred_config, mel_dim, text_dim)
        
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
    
    def get_durpred(self, durpred_config, mel_dim, text_dim):
        spk_encoder = MelStyleEncoder(n_mel_channels=mel_dim, style_vector_dim=self.style_vector_dim)
        durpred = DurationPredictor(
            in_channels = text_dim,
            filter_channels = durpred_config["filter_channels"],
            kernel_size = durpred_config["kernel_size"],
            p_dropout = durpred_config["dropout"],
            style_vector_dim = self.style_vector_dim
        )
        return spk_encoder, durpred
    
    def infer_durpred(self, cond, spk_embed_mask, text_embed, text_len, seq_len):
        spk_embed = self.spk_encoder(cond, spk_embed_mask)
        text_mask = get_mask_from_lengths(text_len, max_len=seq_len)
        text_mask = text_mask.to(text_embed.device)
        logw = self.durpred(text_embed, text_mask.unsqueeze(-1), spk_embed)
        
        w = torch.exp(logw.squeeze(1)) * text_mask
        w_ceil = torch.ceil(w)
        
        mel_lengths = torch.clamp_min(torch.sum(w_ceil, 1), 1).long()
        mel_max_length = mel_lengths.max()
        # Using obtained durations `w` construct alignment map `attn`
        mel_seq_mask = sequence_mask(mel_lengths, mel_max_length).to(text_mask.dtype)
        attn_mask = text_mask.unsqueeze(1) * mel_seq_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.transpose(1,2))
        text_embed = torch.matmul(attn.transpose(1, 2), text_embed)
        
        # clip text_embed to the same len as mel and ppg
        text_embed_len = text_embed.size(1)
        if text_embed_len > seq_len:
            text_embed = text_embed[:, :seq_len, :]
        elif text_embed_len < seq_len:
            text_embed = F.pad(text_embed, (0, 0, 0, seq_len - text_embed_len))

        return text_embed
    
    def train_durpred(self, cond, spk_embed_mask, text_embed, text_len, seq_len, mel, mel_mask):
        spk_embed = self.spk_encoder(cond, spk_embed_mask)
        text_mask = get_mask_from_lengths(text_len, max_len=seq_len)
        text_mask = text_mask.to(text_embed.device)
        logw = self.durpred(text_embed, text_mask.unsqueeze(-1), spk_embed)
        durpred_text_embed = self.durpred_text_embed(text_embed)
        
        text_embed_t = durpred_text_embed.transpose(1, 2) # [b, d, nt]
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
    
        logw_ = torch.log(1e-8 + attn.sum(2)) * text_mask.unsqueeze(1)
        attn = attn.squeeze(1).transpose(1,2)
        dur_loss = duration_loss(logw, logw_, text_len)
        text_embed = torch.matmul(attn.transpose(1, 2), text_embed)
        
        return text_embed, dur_loss
    
    # quantize 10% of txt and ppg tokens
    # text_embed: [b, nt, d], text_len: [b], ppg_embed: [b, n, d], ppg_len: [b]
    def quantize(self, text_embed, ppg_embed, drop_text=False, drop_ppg=False):
        cb_loss = 0
        if not drop_text:
            quantized_text = self.quantizer(text_embed) # quantized_text is a dict
            text_rand_idx = torch.randperm(text_embed.size(1))[:int(text_embed.size(1) * self.codebook_prob)]
            quantized_text_weight = quantized_text["x"].new_zeros(text_embed.size(1))
            quantized_text_weight[text_rand_idx] = 1
            text_embed = quantized_text_weight.unsqueeze(1) * quantized_text["x"] + (1 - quantized_text_weight).unsqueeze(1) * text_embed
            cb_loss += (quantized_text["num_vars"] - quantized_text["prob_perplexity"]) / quantized_text["num_vars"]
        if not drop_ppg:
            quantized_ppg = self.quantizer(ppg_embed)
            ppg_rand_idx = torch.randperm(ppg_embed.size(1))[:int(ppg_embed.size(1) * self.codebook_prob)]
            quantized_ppg_weight = quantized_ppg["x"].new_zeros(ppg_embed.size(1))
            quantized_ppg_weight[ppg_rand_idx] = 1
            ppg_embed = quantized_ppg_weight.unsqueeze(1) * quantized_ppg["x"] + (1 - quantized_ppg_weight).unsqueeze(1) * ppg_embed
            cb_loss += (quantized_ppg["num_vars"] - quantized_ppg["prob_perplexity"]) / quantized_ppg["num_vars"]
        cb_loss *= self.codebook_loss_weight
        return text_embed, ppg_embed, cb_loss
    
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
        spk_embed_mask = None,
        text_len = None,
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

        if self.use_durpred and text is not None:
            text_embed = self.infer_durpred(cond, spk_embed_mask, text_embed, text_len, seq_len)
        
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
        spk_embed_mask = None, # to get ref speech for speaker encoder of duration predictor
        text_len = None,  # text length
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

        if self.use_durpred and text is not None:
            text_embed, dur_loss = self.train_durpred(cond, spk_embed_mask, text_embed, text_len, seq_len, mel, mel_mask)
            extra_loss += dur_loss
        
        if self.use_codebook:
            text_embed, ppg_embed, cb_loss = self.quantize(text_embed, ppg_embed, drop_text=drop_text, drop_ppg=drop_ppg)
            extra_loss += cb_loss
        
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
