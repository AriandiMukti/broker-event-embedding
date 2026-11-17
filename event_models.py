"""Masked Set Transformer -> causal TCN, with optional grouped gates."""
import torch
from torch import nn
import torch.nn.functional as F

from main import Encoder


class SetBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width); self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))

    def forward(self, x, padding):
        attended = self.attention(x, x, x, key_padding_mask=padding, need_weights=False)[0]
        x = self.norm1(x+attended)
        return self.norm2(x+self.ff(x))


class SetEncoder(nn.Module):
    def __init__(self, vocabulary, token_features, width=24, heads=2):
        super().__init__()
        self.identity = nn.Embedding(vocabulary+2, width, padding_idx=0)
        self.project = nn.Linear(token_features, width)
        self.blocks = nn.ModuleList([SetBlock(width, heads) for _ in range(2)])
        self.seed = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.seed, std=.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, tokens, ids, mask):
        b, t, n, f = tokens.shape
        mask = mask.reshape(b*t, n); present = mask.any(1)
        tokens = tokens.reshape(b*t, n, f).masked_fill(~mask[..., None], 0)
        ids = ids.reshape(b*t, n).masked_fill(~mask, 0)
        x = self.project(tokens)+self.identity(ids)
        # Empty sets use one internal sentinel to avoid all-masked softmax NaN.
        safe = mask.clone(); safe[~present, 0] = True
        for block in self.blocks:
            x = block(x, ~safe)
        pooled = self.pool(self.seed.expand(b*t, -1, -1), x, x,
                           key_padding_mask=~safe, need_weights=False)[0][:, 0]
        return (pooled*present[:, None]).reshape(b, t, -1)


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.left = 2*dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        y = self.conv(F.pad(x, (self.left, 0)))
        return self.norm((x+F.gelu(y)).transpose(1, 2)).transpose(1, 2)


class EventModel(nn.Module):
    def __init__(self, features, token_features, vocabulary, groups, mode='none', width=24):
        super().__init__()
        if mode not in ['none', 'static', 'dynamic']:
            raise ValueError(mode)
        self.groups, self.group_names, self.mode = groups, list(groups), mode
        assignment = torch.zeros(features, len(groups))
        for j, ids in enumerate(groups.values()):
            assignment[ids, j] = 1
        if not torch.all(assignment.sum(1) == 1):
            raise ValueError('Each daily feature must belong to exactly one gate group')
        self.register_buffer('assignment', assignment)
        self.set_encoder = SetEncoder(vocabulary, token_features, width)
        self.context = nn.Linear(features, width)
        if mode == 'static':
            self.logits = nn.Parameter(torch.zeros(len(groups)))
        elif mode == 'dynamic':
            self.controller = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, len(groups)))
            nn.init.zeros_(self.controller[-1].weight); nn.init.zeros_(self.controller[-1].bias)
        self.fusion = nn.Linear(2*width, width)
        self.temporal = nn.Sequential(*[CausalBlock(width, d) for d in [1, 2, 4]])
        self.head = nn.Linear(width, 9)  # 3 return, 3 log-vol, 3 direction logits

    def forward(self, x, tokens, ids, mask):
        if self.mode == 'static':
            gates = self.logits.sigmoid().expand(*x.shape[:2], -1)
        elif self.mode == 'dynamic':
            gates = self.controller(x).sigmoid()
        else:
            gates = x.new_ones(*x.shape[:2], len(self.groups))
        gated = x*(gates @ self.assignment.T)
        broker = self.set_encoder(tokens, ids, mask)
        broker = broker*gates[:, :, self.group_names.index('broker'), None]
        z = self.fusion(torch.cat([broker, self.context(gated)], -1))
        temporal = self.temporal(z.transpose(1, 2))
        return self.head(temporal[:, :, -1]), gates


class LegacyCNN(nn.Module):
    """Reuse main.Encoder unchanged, attach probability/log-vol auxiliary heads.

    Reuses the existing CNN/embedding, with the same multitask supervised loss
    as the new variants for a matched comparison. The unused original decoder
    is preserved; old reconstruction-trained checkpoints are separate evidence.
    """
    def __init__(self, features, window, vocabulary, token_features):
        super().__init__()
        self.brokers = vocabulary+2
        self.encoder = Encoder(features+self.brokers*token_features, window, channels=16, compact_decoder=True)
        self.auxiliary = nn.Linear(32, 6)

    def prepare_input(self, x, tokens, ids, mask):
        b, t, n, f = tokens.shape
        flat = tokens.new_zeros(b, t, self.brokers, f)
        flat.scatter_add_(2, ids[..., None].expand(-1, -1, -1, f), tokens*mask[..., None])
        return torch.cat([x, flat.flatten(2)], -1)

    def forward(self, x, tokens=None, ids=None, mask=None):
        z, returns, _ = self.encoder(self.prepare_input(x, tokens, ids, mask))
        aux = self.auxiliary(z)
        return torch.cat([returns, aux[:, :3], aux[:, 3:]], -1), x.new_empty(len(x), x.shape[1], 0)
