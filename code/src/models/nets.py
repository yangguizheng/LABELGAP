"""Deep-learning baselines B2-B5. Deliberately plain: no attention modules,
no CBAM/SE, no custom losses. All take the same preprocessed vector x in R^d."""
from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.head(self.body(x))


class CNN1D(nn.Module):
    """x is read as a 1-channel sequence of length d (feature order fixed)."""

    def __init__(self, in_dim: int, n_classes: int, channels=(32, 64), kernel_size=3,
                 fc_dim=128, dropout=0.3):
        super().__init__()
        layers, c_in, length = [], 1, in_dim
        for c in channels:
            layers += [nn.Conv1d(c_in, c, kernel_size, padding=kernel_size // 2),
                       nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]
            c_in, length = c, length // 2
        self.features = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c_in * length, fc_dim), nn.ReLU(),
                                nn.Dropout(dropout))
        self.head = nn.Linear(fc_dim, n_classes)

    def forward(self, x):
        return self.head(self.fc(self.features(x.unsqueeze(1))))


class BiGRU(nn.Module):
    """x is read as a sequence of d steps with one scalar per step; the final
    hidden states of both directions of the last layer are concatenated."""

    def __init__(self, in_dim: int, n_classes: int, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        _, h = self.gru(x.unsqueeze(-1))
        return self.head(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


class FeatureTokenizer(nn.Module):
    """Per-feature linear embedding x_j * W_j + b_j (FT-Transformer tokenizer,
    Gorishniy et al., NeurIPS 2021)."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        bound = 1.0 / math.sqrt(d_token)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class TransformerClassifier(nn.Module):
    """Standard nn.TransformerEncoder over feature tokens + [CLS] token."""

    def __init__(self, in_dim: int, n_classes: int, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, norm_first=True, activation="relu"):
        super().__init__()
        self.tokenizer = FeatureTokenizer(in_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           activation=activation, batch_first=True,
                                           norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        tokens = self.tokenizer(x)
        tokens = torch.cat([self.cls.expand(x.size(0), -1, -1), tokens], dim=1)
        z = self.encoder(tokens)
        return self.head(self.norm(z[:, 0]))


BUILDERS = {"mlp": MLP, "cnn": CNN1D, "bigru": BiGRU, "transformer": TransformerClassifier}


def build_model(name: str, in_dim: int, n_classes: int, params: dict) -> nn.Module:
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
    return BUILDERS[name](in_dim, n_classes, **kwargs)


def gru_flops(gru: nn.GRU, seq_len: int) -> int:
    """Forward FLOPs of an nn.GRU (multiply-add = 2 FLOPs; gate matmuls only)."""
    dirs = 2 if gru.bidirectional else 1
    h, total = gru.hidden_size, 0
    for layer in range(gru.num_layers):
        in_l = gru.input_size if layer == 0 else h * dirs
        total += dirs * seq_len * 2 * 3 * h * (in_l + h)
    return total


def count_flops(model: nn.Module, in_dim: int, device) -> int:
    """Forward FLOPs for one sample. torch's FlopCounterMode has no formula for
    cuDNN RNN kernels, so GRU layers are added analytically when uncounted."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    x = torch.zeros(1, in_dim, device=device)
    counter = FlopCounterMode(display=False)
    # grad enabled -> nn.TransformerEncoderLayer skips its fused fast path,
    # whose kernel the counter cannot see.
    with torch.enable_grad(), counter:
        model(x)
    total = counter.get_total_flops()
    global_ops = counter.get_flop_counts().get("Global", {})
    rnn_counted = any("rnn" in str(op).lower() or "gru" in str(op).lower() for op in global_ops)
    if not rnn_counted:
        total += sum(gru_flops(m, seq_len=in_dim) for m in model.modules()
                     if isinstance(m, nn.GRU))
    model.train(was_training)
    return int(total)
