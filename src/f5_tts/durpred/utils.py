import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import namedtuple

class LayerNorm(nn.Module):
    def __init__(
        self, 
        dim, 
        bias=True, 
        eps=1e-5
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv1dGLU(nn.Module):
    '''
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    '''

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=kernel_size, padding=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x

def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape

class SpkEncoderSelfAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    self.attn = None

    self.k_channels = channels // n_heads
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    nn.init.xavier_uniform_(self.conv_v.weight)
    if proximal_init:
      with torch.no_grad():
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)
      
  def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
    
    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    x = self.conv_o(x)
    return x

  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query /math.sqrt(self.k_channels), key_relative_embeddings)
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores.masked_fill(block_mask == 0, -1e4)
    p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    max_relative_position = 2 * self.window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start_position = max((self.window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final

  def _attention_bias_proximal(self, length):
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


def mask_from_start_end_indices(seq_len, start, end):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len, frac_lengths):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def random_masking(mask, mask_prob):
    assert mask.ndim == 2
    lens = mask.shape[-1]
    rand = torch.randn(mask.shape, device=mask.device)
    rand[:, 0] = -torch.finfo(rand.dtype).max # Ignore the first item
    num_mask = min(int(lens * mask_prob), lens - 1)
    indices = rand.topk(num_mask, dim=-1).indices
    new_mask = ~torch.zeros(mask.shape, device=mask.device).scatter(1, indices, 1.).bool()
    return new_mask


# 把一个 batch 长短不一的音频中，每一段音频比最长音频更短，即没有声音的那一段 mask 掉
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


class EmbeddingTable(nn.Embedding):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        pad_id=-1,
        **kwargs
    ):
        super().__init__(
            num_embeddings, 
            embedding_dim,
            **kwargs
        )
        
        nn.init.normal_(self.weight, 0.0, embedding_dim ** -0.5)
        self.pad_id = pad_id
        self.output_dim = embedding_dim

    def forward(self, x):
        if self.pad_id is not None:
            mask = x == self.pad_id
            x = x.masked_fill(mask, 0)
        outputs = super().forward(x)
        if self.pad_id is not None:
            outputs = outputs.masked_fill(mask.unsqueeze(-1), 0.)
        return outputs


class Linear(nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        w_init_gain: str = 'linear',
        activation = None,
        **kwargs
    ):
        super(Linear, self).__init__(
            in_channels,
            out_channels,
            bias=bias
        )

        self.activation = activation if activation is not None else nn.Identity()
        self.output_dim = out_channels
        if w_init_gain is not None:
            if isinstance(w_init_gain, str):
                gain = nn.init.calculate_gain(w_init_gain)
            else:
                gain = w_init_gain
            nn.init.xavier_uniform_(
                    self.weight, gain=gain)
        if bias:
            nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x, **kwargs):
        return self.activation(super(Linear, self).forward(x))

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        bias: bool = True,
        padding = None,
        causal: bool = False,
        bn: bool = False,
        activation = None,
        w_init_gain = None,
        input_transpose: bool = False,
        **kwargs
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias
        )

        self.in_channels = in_channels
        self.transpose = input_transpose
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        if w_init_gain is not None:
            nn.init.xavier_uniform_(
                self.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        outputs = self.activation(self.bn(super(Conv1d, self).forward(x)))
        return outputs.transpose(1, 2) if self.transpose else outputs

    def extra_repr(self):
        return '(settings): {}\n(causal): {}\n(input_transpose): {}'.format(
                super(Conv1d, self).extra_repr(), self.causal, self.transpose)

class ConvPositionEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 31,
        groups: int = 16,
        causal:bool = False
    ):
        super().__init__()
        self.conv = Conv1d(
            hidden_size, 
            hidden_size, 
            kernel_size, 
            groups=groups,
            input_transpose=True,
            activation=nn.GELU(),
            causal=causal)

    def forward(self, x, mask=None):

        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.)

        x = self.conv(x)
        if mask is not None:
            x = x.masked_fill(~mask, 0.)

        return x
    

class Dropout(nn.Module):
    def __init__(
        self, 
        p: float = 0.5, 
        inplace: bool = False,
        force_drop: bool = False,
        **kwargs
    ):
        super(Dropout, self).__init__() 
        if p < 0. or p > 1.:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.force_drop = force_drop

    def forward(self, x, **kwargs):
        return F.dropout(
            x, 
            p=self.p, 
            training=True if self.force_drop else self.training,
            inplace=self.inplace
        )

    def extra_repr(self):
        return 'prob={}, inplace={}, force_drop={}'.format(
                self.p, self.inplace, self.force_drop)
    

class Mlp(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            ffn_hidden_size=4096, 
            act_layer=nn.GELU, 
            dropout=0.,
            **kwargs
    ):
        super().__init__()
        self.fc1 = Linear(hidden_size, ffn_hidden_size)
        self.act = act_layer()
        self.fc2 = Linear(ffn_hidden_size, hidden_size)
        self.drop = Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = Linear(hidden_size, output_size, bias=True)

    def forward(self, x, c, mask=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x
    
    
class TransformerBlock(nn.Module):
    """Conditional transformer block"""
    def __init__(
            self, 
            attention: nn.Module,
            ffn: nn.Module,
            hidden_size: int = 1024,
            modulation: bool = False,
            eps: float = 1e-6,
            **kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=not modulation, eps=eps)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=not modulation, eps=eps)
        self.attn = attention # Attention block instance
        self.ffn = ffn # Feed-forward block instance
        self.modulation = modulation
        if modulation:
            self.modulation_layer = nn.Sequential(
                nn.SiLU(),
                Linear(hidden_size, 6 * hidden_size, bias=True)
            )
            # Zero-init from DiT
            nn.init.constant_(self.modulation_layer[-1].weight, 0.)
            nn.init.constant_(self.modulation_layer[-1].bias, 0.)

    def forward(self, x, condition=None, mask=None,pos_ids=None,cache=None):
        if condition is None:
            assert not self.modulation, "Without global condition, must set modulation to False"
        else:
            assert self.modulation, "With global condition, must set modulation to True"
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation_layer(condition).chunk(6, dim=1)

        # Attention forward
        if condition is not None:
            x = x + gate_attn.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_attn, scale_attn), mask=mask,pos_ids=pos_ids,cache=cache)
        else:
            x = x + self.attn(self.norm1(x), mask=mask,pos_ids=pos_ids,cache=cache)

        # FFN forward
        if condition is not None:
            x = x + gate_ffn.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        else:
            x = x + self.ffn(self.norm2(x), mask=mask)
        return x
    

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


class AlibiEmbedding(nn.Module):
    """ Symmetric version of Alibi"""
    def __init__(self, num_heads, max_position_embeddings=1024):
        super().__init__()
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.register_buffer("alibi", self.build_alibi_tensor())

    @property
    def device(self):
        return self.alibi.device

    def build_alibi_tensor(self):
        # For simplicity, compute the symmetric distance among all tokens,
        # thus the mask is important for causal modeling (Shan)
        context_position = torch.arange(self.max_position_embeddings)[:, None]
        memory_position = torch.arange(self.max_position_embeddings)[None, :]
        relative_position = memory_position - context_position 
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.num_heads, -1,-1)
        slopes = torch.Tensor(get_slopes(self.num_heads)) * -1
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        alibi = alibi.view(1, self.num_heads, self.max_position_embeddings, self.max_position_embeddings)
        return alibi

    @torch.autocast(enabled = False, device_type="cuda")
    def forward(self, L, S):
        max_len = max(L, S)
        # Update
        if max_len > self.max_position_embeddings:
            print(f"Updating alibi matrix for {self.device}")
            self.max_position_embeddings = max_len
            self.register_buffer("alibi", self.build_alibi_tensor())
        return self.alibi[:, :, :L, :S]


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @torch.autocast(enabled = False, device_type="cuda")
    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.arange(t, device = self.device)
        t = t.type_as(self.inv_freq)
        if t.dim() == 1:
            freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        else:
            freqs = torch.einsum('bi, j -> bij', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs
    

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)


@torch.autocast(enabled = False, device_type="cuda")
def apply_rotary_pos_emb(pos, t):
    if pos.dim()==3:
        pos = pos.unsqueeze(1)
    return t * pos.cos() + rotate_half(t) * pos.sin()


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    def __init__(
            self,
            hidden_size: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            dropout: float = 0.0,
            max_position_embeddings: int = 4096,
            norm_layer: nn.Module = nn.LayerNorm,
            alibi_bias: bool = False,
            rotary_bias: bool = False,
            name:str=None,
            **kwargs
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, 'hidden_size should be divisible by num_heads'
        assert not( alibi_bias and rotary_bias), 'alibi_bias rotary_bias cannot be all True'
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_position_embeddings = max_position_embeddings
        self.alibi_bias = alibi_bias
        self.rotary_bias = rotary_bias 

        self.q_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = Dropout(attn_drop)
        self.o_proj = Linear(hidden_size, hidden_size)
        self.o_dropout = Dropout(dropout)

        self.cpu_config = AttentionConfig(True, True, True)
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            # TODO, should be True, False, False, but got error in my docker images
            self.cuda_config = AttentionConfig(True, True, True)
        else:
            self.cuda_config = AttentionConfig(False, True, True)

        if self.alibi_bias: 
            self.alibi = AlibiEmbedding(self.num_heads)

        if self.rotary_bias: 
            self.rotary = RotaryEmbedding(self.head_dim)
        self.name = name

    # @torch.autocast(device_type="cuda")
    def forward(self, q, k=None, v=None, mask=None,pos_ids=None,cache=None):
        if cache is None:
            k = k or q
            v = v or q
            B, L, C = q.shape
            B, S, C = v.shape
            if mask is not None:
                if mask.ndim == 2: # [B, L]
                    assert L == S
                    mask = rearrange(mask, 'b j -> b 1 1 j')
                    mask = mask.expand(-1, self.num_heads, L, -1)
                elif mask.ndim == 3: # [B, L, S]
                    assert mask.size(1) == L and mask.size(2) == S
                    mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
            q = rearrange(q, 'b n (h d) -> b h n d', h = self.num_heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h = self.num_heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h = self.num_heads)
            q, k = self.q_norm(q), self.k_norm(k)

            config = self.cuda_config if q.is_cuda else self.cpu_config
            attn_bias = torch.zeros(B, self.num_heads, L, S, dtype=q.dtype, device=q.device)

            # Apply alibi
            if self.alibi_bias:
                attn_bias += self.alibi(L, S)

            # Apply rotary
            if self.rotary_bias:
                if L == S:
                    if pos_ids is None:
                        rotary_emb = self.rotary(L)
                    else:
                        rotary_emb = self.rotary(pos_ids)
                    q, k = map(lambda x: apply_rotary_pos_emb(rotary_emb, x), (q, k))
                else:
                    q_rotary_emb = self.rotary(L)
                    k_rotary_emb = self.rotary(S)
                    q = apply_rotary_pos_emb(q_rotary_emb, q)
                    k = apply_rotary_pos_emb(k_rotary_emb, k)

            if mask is not None:
                attn_bias.masked_fill_(mask.logical_not(), float("-inf"))


            with torch.backends.cuda.sdp_kernel(**config._asdict()):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_bias,
                    dropout_p=self.attn_drop.p if self.training else 0.)
            
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.o_dropout(self.o_proj(out))
            return out
        else:
            chunk_size = 32
            # raise NotImplementedError("Cache is not implemented for MultiHeadAttention")
            # pdb.set_trace()
            k = k or q
            v = v or q
            B, L, C = q.shape
            B, S, C = v.shape
            is_cached = cache[self.name]["k"] is not None
            # 使用缓存来存储和更新k和v
            if is_cached:
                # pdb.set_trace()
                # 从缓存中获取之前的k和v
                cached_k, cached_v = cache[self.name]['k'], cache[self.name]['v']
                # 仅对新的部分k和v进行计算
                k = torch.cat([cached_k, self.k_proj(k[:, cached_k.shape[1]:, :])], dim=1)
                v = torch.cat([cached_v, self.v_proj(v[:, cached_v.shape[1]:, :])], dim=1)
            else:
                # 第一次调用，计算全部的k和v
                k = self.k_proj(k)
                v = self.v_proj(v)
            # 更新缓存
            cache[self.name]['k'] = k[:,:-chunk_size, :]
            cache[self.name]['v'] = v[:,:-chunk_size, :]

            if is_cached:
                output_cached = cache[self.name]["o"]
            
            if is_cached:
                q = q[:,-2 * chunk_size:, :]
            q = self.q_proj(q)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
            q, k = self.q_norm(q), self.k_norm(k)
            config = self.cuda_config if q.is_cuda else self.cpu_config
            attn_bias = torch.zeros(B, self.num_heads, L, S, dtype=q.dtype, device=q.device)
            if self.rotary_bias:
                if L == S:
                    if pos_ids is None:
                        rotary_emb = self.rotary(L)
                    else:
                        rotary_emb = self.rotary(pos_ids) # torch.Size([1, 600, 64])
                    if is_cached:
                        q = apply_rotary_pos_emb(rotary_emb[:,-2*chunk_size:], q)
                        k = apply_rotary_pos_emb(rotary_emb, k)
                    else:
                        q, k = map(lambda x: apply_rotary_pos_emb(rotary_emb, x), (q, k))

                else:
                    q_rotary_emb = self.rotary(L)
                    k_rotary_emb = self.rotary(S)
                    q = apply_rotary_pos_emb(q_rotary_emb, q)
                    k = apply_rotary_pos_emb(k_rotary_emb, k)
            if mask is not None:
                attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            if is_cached:
                # pdb.set_trace()
                # q = q[:,:, -2:, :]
                attn_bias = attn_bias[:, :, -2*chunk_size:, :]
            # with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.) 
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.o_dropout(self.o_proj(out))
            
            if is_cached:
                # pdb.set_trace()
                out = torch.cat([output_cached, out], dim=1)
                
                cache[self.name]["o"] = out[:, :-chunk_size, :]
            else:
                cache[self.name]["o"] = out[:, :-chunk_size, :]
            return out
        
def mle_loss(z, m, logs, mask, logdet=None):
  l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m) ** 2)) # neg normal likelihood w/o the constant term
  if logdet is not None:
    l = l - torch.sum(logdet) # log jacobian determinant
  l = l / torch.sum(torch.ones_like(z) * mask) # averaging across batch, channel and time axes
  l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
  return l

def duration_loss(logw, logw_, lengths):
  l = torch.sum((logw - logw_)**2) / torch.sum(lengths)
  return l


# 混元 duration predictor 的 speaker encoder
class SpkEncoder(torch.nn.Module):
    def __init__(self, in_dim=100, hidden_dim=128, out_dim=256):
        super().__init__()

        self.in_dim = in_dim # Linear 513 wav2vec 2.0 1024
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.1

        self.spectral = nn.Sequential(
            nn.Conv1d(self.in_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = SpkEncoderSelfAttention(self.hidden_dim, self.hidden_dim, self.n_head, p_dropout = self.dropout, proximal_bias= False, proximal_init=True)
        self.atten_drop = nn.Dropout(self.dropout)
        self.fc = nn.Conv1d(self.hidden_dim, self.out_dim, 1)

    # x: [batch, duration, hidden_dim], mask: [batch, duration]
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.full(x.shape[:2], True).cuda()
        x = x.transpose(1, 2)
        mask = mask.unsqueeze(1)

        x = self.spectral(x) * mask # spectral        
        x = self.temporal(x) * mask # temporal

        # self-attention
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        y = self.slf_attn(x,x, attn_mask=attn_mask)
        x = x+ self.atten_drop(y)

        # fc
        x = self.fc(x)

        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=2)
        else:
            len_ = mask.sum(dim=2)
            x = x.sum(dim=2)

            out = torch.div(x, len_)
        return out
    
    
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
    