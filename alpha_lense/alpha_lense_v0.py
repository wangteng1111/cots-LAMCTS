"""Alpha Lense v0: physics-grounded search-driven policy/value learning.

Core idea:
  optical prescription -> Transformer -> policy/value/merit
  policy -> physical adjustment targets -> COTS projection
  MCTS + authoritative physics evaluator -> improved policy/value targets
  replay -> Transformer update

The physics evaluator is the sole authority for realised merit.  The network
never replaces it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence
import math
import random
import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor

@dataclass(frozen=True)
class Candidate:
    parts: tuple[int, ...]

@dataclass
class EvalResult:
    J: float
    metrics: dict[str, float]

@dataclass
class SearchTarget:
    state: Candidate
    surface_tokens: Tensor
    mask: Tensor
    policy: Tensor
    value: float
    immediate_merit: float

class Catalog(Protocol):
    def legal_actions(self, state: Candidate) -> Sequence[int]: ...
    def apply(self, state: Candidate, action: int) -> Candidate: ...
    def action_features(self, state: Candidate, actions: Sequence[int]) -> Tensor: ...

class AlphaLenseNet(nn.Module):
    """Prescription encoder with merit, reachable-value and adjustment-policy heads."""
    def __init__(self, token_dim=8, d_model=64, nhead=4, layers=3, ff=192,
                 action_dim=32, max_surfaces=64):
        super().__init__()
        self.proj = nn.Linear(token_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_surfaces + 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.merit = nn.Sequential(nn.Linear(d_model, 48), nn.GELU(), nn.Linear(48, 1))
        self.value = nn.Sequential(nn.Linear(d_model, 48), nn.GELU(), nn.Linear(48, 1))
        self.adjust = nn.Linear(d_model, action_dim)
        self.action_key = nn.Linear(action_dim, d_model, bias=False)

    def encode(self, tokens: Tensor, mask: Tensor) -> Tensor:
        b, n, _ = tokens.shape
        x = self.proj(tokens)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], 1) + self.pos[:, :n+1]
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=mask.device), ~mask.bool()], 1)
        return self.norm(self.encoder(x, src_key_padding_mask=pad)[:, 0])

    def forward(self, tokens: Tensor, mask: Tensor, action_features: Tensor | None = None):
        h = self.encode(tokens, mask)
        out = {"embedding": h, "merit": self.merit(h).squeeze(-1),
               "value": self.value(h).squeeze(-1), "adjustment": self.adjust(h)}
        if action_features is not None:
            # action_features: [B,A,action_dim]; learned physical action compatibility.
            keys = self.action_key(action_features)
            out["policy_logits"] = torch.einsum("bd,bad->ba", h, keys) / math.sqrt(h.shape[-1])
        return out

@dataclass
class Node:
    state: Candidate
    prior: Tensor
    actions: Sequence[int]
    N: Tensor
    W: Tensor
    children: dict[int, "Node"]
    expanded: bool = False
    immediate_merit: float | None = None

    @property
    def Q(self):
        return torch.where(self.N > 0, self.W / self.N.clamp_min(1), torch.zeros_like(self.W))

class AlphaLenseMCTS:
    """PUCT search. Leaf truth may be NN value; realised candidates are grounded by physics."""
    def __init__(self, net: AlphaLenseNet, catalog: Catalog,
                 tokenize: Callable[[Candidate], tuple[Tensor, Tensor]],
                 physics: Callable[[Candidate], EvalResult], c_puct=1.5, device="cuda"):
        self.net, self.catalog, self.tokenize, self.physics = net, catalog, tokenize, physics
        self.c_puct, self.device = c_puct, device
        self.physics_cache: dict[Candidate, EvalResult] = {}

    @torch.no_grad()
    def expand(self, node: Node) -> float:
        actions = list(self.catalog.legal_actions(node.state))
        tok, mask = self.tokenize(node.state)
        af = self.catalog.action_features(node.state, actions)
        out = self.net(tok[None].to(self.device), mask[None].to(self.device), af[None].to(self.device))
        prior = F.softmax(out["policy_logits"][0], -1).cpu()
        node.actions, node.prior = actions, prior
        node.N = torch.zeros(len(actions)); node.W = torch.zeros(len(actions)); node.children = {}
        node.expanded = True
        return float(out["value"].item())

    def search(self, root_state: Candidate, simulations=256, physics_topk=8, temperature=1.0):
        root = Node(root_state, torch.empty(0), [], torch.empty(0), torch.empty(0), {})
        self.expand(root)
        for _ in range(simulations):
            node, path = root, []
            while node.expanded and len(node.actions):
                score = node.Q + self.c_puct * node.prior * math.sqrt(float(node.N.sum()) + 1.0) / (1.0 + node.N)
                i = int(torch.argmax(score))
                path.append((node, i))
                if i not in node.children:
                    s2 = self.catalog.apply(node.state, node.actions[i])
                    node.children[i] = Node(s2, torch.empty(0), [], torch.empty(0), torch.empty(0), {})
                node = node.children[i]
                if not node.expanded: break
            v = self.expand(node)
            for parent, i in reversed(path):
                parent.N[i] += 1; parent.W[i] += v

        # Physics-ground the most visited root actions.  Convert lower J to larger value -log(J).
        order = torch.argsort(root.N, descending=True)[:min(physics_topk, len(root.actions))]
        for idx in order.tolist():
            child = root.children.get(idx)
            if child is None:
                s2 = self.catalog.apply(root.state, root.actions[idx])
                child = Node(s2, torch.empty(0), [], torch.empty(0), torch.empty(0), {})
                root.children[idx] = child
            if child.state not in self.physics_cache:
                self.physics_cache[child.state] = self.physics(child.state)
            z = -math.log(max(self.physics_cache[child.state].J, 1e-12))
            # Authoritative physics backup at root.
            root.N[idx] += 1; root.W[idx] += z

        visits = root.N.pow(1.0 / max(temperature, 1e-6))
        pi = visits / visits.sum().clamp_min(1e-12)
        best_i = int(torch.argmax(root.N))
        next_state = self.catalog.apply(root.state, root.actions[best_i])
        return next_state, pi, root

class ReplayBuffer:
    def __init__(self, capacity=200_000): self.capacity, self.data = capacity, []
    def add(self, x: SearchTarget):
        self.data.append(x)
        if len(self.data) > self.capacity: del self.data[:len(self.data)-self.capacity]
    def sample(self, n): return random.sample(self.data, min(n, len(self.data)))

def train_step(net: AlphaLenseNet, optimizer, batch: Sequence[SearchTarget],
               catalog: Catalog, device="cuda", value_w=1.0, merit_w=0.5, policy_w=1.0):
    """AlphaZero-like loss: MCTS policy + search value + dense immediate physics merit."""
    tokens = torch.stack([b.surface_tokens for b in batch]).to(device)
    masks = torch.stack([b.mask for b in batch]).to(device)
    actions = [list(catalog.legal_actions(b.state)) for b in batch]
    if len({len(a) for a in actions}) != 1:
        raise ValueError("v0 trainer expects equal legal-action counts per minibatch; bucket by action count")
    af = torch.stack([catalog.action_features(b.state, a) for b, a in zip(batch, actions)]).to(device)
    out = net(tokens, masks, af)
    pi = torch.stack([b.policy for b in batch]).to(device)
    z = torch.tensor([b.value for b in batch], dtype=torch.float32, device=device)
    q = torch.tensor([b.immediate_merit for b in batch], dtype=torch.float32, device=device)
    lp = F.log_softmax(out["policy_logits"], -1)
    l_policy = -(pi * lp).sum(-1).mean()
    l_value = F.mse_loss(out["value"], z)
    l_merit = F.mse_loss(out["merit"], q)
    loss = policy_w*l_policy + value_w*l_value + merit_w*l_merit
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return {"loss": float(loss.detach()), "policy": float(l_policy.detach()),
            "value": float(l_value.detach()), "merit": float(l_merit.detach())}

# Integration contract:
# 1) tokenize(state) MUST emit the existing OPT surface representation (<=64 x 8).
# 2) Catalog actions MUST be physical substitutions/adjustments, not arbitrary ID semantics.
# 3) physics(state) MUST call the authoritative production evaluator (Q4096 in current project).
# 4) Search targets: pi from visit counts; value from best physics-grounded descendant;
#    immediate_merit = -log(J(state)).
# 5) For real COTS deployment, Catalog projects adjustment/action features onto vendor+SKU parts.
