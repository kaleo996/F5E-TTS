import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import namedtuple


# when using durpred, add padding between every two phoneme tokens to model the pause / blurry part between
def intersperse(text, sep=" ") -> list[list[str]]:
    output = []
    for sentence in text:
        interspersed_sentence = [sep] * (len(sentence) * 2 + 1)
        interspersed_sentence[1::2] = sentence
        output.append(interspersed_sentence)
    return output


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype, device=duration.device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path.float()
    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    path = path * mask
    return path


def random_masking(mask, mask_prob):
    assert mask.ndim == 2
    lens = mask.shape[-1]
    rand = torch.randn(mask.shape, device=mask.device)
    rand[:, 0] = -torch.finfo(rand.dtype).max # Ignore the first item
    num_mask = min(int(lens * mask_prob), lens - 1)
    indices = rand.topk(num_mask, dim=-1).indices
    new_mask = ~torch.zeros(mask.shape, device=mask.device).scatter(1, indices, 1.).bool()
    return new_mask


# for a batch of mel, mask out the padding part of each mel
def get_mask_from_lengths(lengths, max_len=None, r=1, random_mask=0.):
    if max_len is None:
        max_len = torch.max(lengths).item()
    if max_len % r != 0:
        max_len = max_len + r - max_len % r
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(lengths.device))
    mask = (ids < lengths.unsqueeze(1)).bool()
    if random_mask > 0.:
        mask = mask.logical_and(random_masking(mask, random_mask))
    return mask


# convert a list of tensors in different lens to a tensor of shape (batch_size, max_len, dim)
def list2tensor(embeds):
    max_len = max([e.shape[0] for e in embeds])
    tensors = []
    for e in embeds:
        pad_len = max_len - e.shape[0]
        if pad_len > 0:
            e = F.pad(e, (0, 0, 0, pad_len), value=0)
        tensors.append(e)
    return torch.stack(tensors)


def duration_loss(logw, logw_, lengths):
  l = torch.sum((logw - logw_)**2) / torch.sum(lengths)
  return l


class Conv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x
    