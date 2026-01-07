from tarfile import OutsideDestinationError
from this import d
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight.T
        return x

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeds = num_embeddings
        self.embed_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std =0.02)

    def forward(self,token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        x = x / norm
        x = x * self.weight
        return x.to(in_dtype)


class swiglu(nn.Module):
    def __init__(self, d_model: int, dff: int, w1, w2, w3, device=None, dtype=None):
        super().__init__()
        self.d = d_model
        #dff = math.ceil((8 * d_model) / 64) * 64
        self.dff = dff

        self.weight1 = w1
        self.weight2 = w2
        self.weight3 = w3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w1 = x @ self.weight1.T
        x_w3 = x @ self.weight3.T 
        gated = x_w1 * torch.sigmoid(x_w1) * x_w3
        # SiLU = W1_x * sigmoid(W1_x)
        # SwiGLU = W2 * SiLU * W3_x
        return gated @ self.weight2.T

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE: d_k must be even")
        self.theta = theta
        self.d_k = d_k
        self.l = max_seq_len
        self.device = device
        self.dtype = dtype

        inv_freq = theta ** (-(torch.arange(0, d_k, 2, device=device, dtype=torch.float32))/d_k)
        pos = torch.arange(self.l, device=device).float()
        angles = pos[:, None] * inv_freq[None, :] # shape: (l, d_k/2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        print(cos.shape, sin.shape) 
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

# 3.5.4 Scaled Dot-Product Attention: Softmax
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x-x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scale_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = q.shape[-1]
    if d_k != k.shape[-1]:
        raise ValueError("d_k must be the same for q, k, and v")

    k = k.transpose(-2, -1)

    x = q @ k / (d_k ** 0.5)
    if mask is not None:
        x = x.masked_fill(~mask, float("-inf"))
    x = softmax(x, dim=-1)
    x = x @ v

    return x

# 3.5.5 Causal Multi-Head Self-Attention
class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device= None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.nh = num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_head = d_model // num_heads
        self.device =device
        self.dtype = dtype

        self.w_q = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_k = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_v = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_o = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.w_q, std=0.02)
        nn.init.trunc_normal_(self.w_k, std=0.02)
        nn.init.trunc_normal_(self.w_v, std=0.02)
        nn.init.trunc_normal_(self.w_o, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = x @ self.w_q.T
        k = x @ self.w_k.T
        v = x @ self.w_v.T

        q = q.view(B, T, self.nh, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.nh, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.nh, self.d_head).transpose(1, 2)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))[None, None, :, :]

        attn = scale_qkv(q, k, v, causal_mask)
        out = attn.transpose(1,2).contiguous()
        out = out.view(B, T, self.d_model)

        out = out @ self.w_o.T
        return out


class MHA_rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device= None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.nh = num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_head = d_model // num_heads
        self.device =device
        self.dtype = dtype

        self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device= device, dtype=dtype)


        self.w_q = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_k = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_v = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_o = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.w_q, std=0.02)
        nn.init.trunc_normal_(self.w_k, std=0.02)
        nn.init.trunc_normal_(self.w_v, std=0.02)
        nn.init.trunc_normal_(self.w_o, std=0.02)

    def forward(self, x: torch.Tensor, token_positions) -> torch.Tensor:
        B, T, _ = x.shape

        q = x @ self.w_q.T
        k = x @ self.w_k.T
        v = x @ self.w_v.T

        q = q.view(B, T, self.nh, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.nh, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.nh, self.d_head).transpose(1, 2)

        q_rope = self.rope(q, token_positions)
        k_rope = self.rope(k, token_positions)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))[None, None, :, :]

        attn = scale_qkv(q_rope, k_rope, v, causal_mask)
        out = attn.transpose(1,2).contiguous()
        out = out.view(B, T, self.d_model)

        out = out @ self.w_o.T
        return out

# 3.6 The Full Transformer LM
class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff:int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.nh = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.attn = MHA_rope(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:



