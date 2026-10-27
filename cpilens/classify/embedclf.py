"""Classifiers on frozen sentence embeddings, mirroring Spectrum: k-NN, linear, small MLP.

All return logits over a fixed class list so one temperature-scaling step calibrates any of
them. k-NN logits are log similarity-weighted vote shares (with a small floor). The linear
model and the MLP share one PyTorch trainer: Adam, class weights proportional to
1/sqrt(class frequency) (per-label recall drives the CPI-weighted metric, so rare labels are
up-weighted, but not fully), early stopping on validation loss. Validation is checked every
max(one pass, 50 steps) with batch <= n/4, so small training sets are still trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLOOR = 1e-4


def encode_labels(y: list[str], classes: list[str]) -> np.ndarray:
    pos = {c: i for i, c in enumerate(classes)}
    return np.array([pos[v] for v in y], dtype=np.int64)


# ---------------------------------------------------------------- k-NN


def knn_logits(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, k: int, n_classes: int, chunk: int = 4096
) -> np.ndarray:
    """Cosine k-NN (rows are L2-normalised), votes weighted by similarity."""
    a = torch.from_numpy(xtr).to(DEVICE)
    lab = torch.from_numpy(ytr).to(DEVICE)
    out = []
    for s in range(0, len(xte), chunk):
        b = torch.from_numpy(xte[s : s + chunk]).to(DEVICE)
        sim, idx = (b @ a.T).topk(min(k, len(a)), dim=1)
        votes = torch.zeros(len(b), n_classes, device=DEVICE)
        votes.scatter_add_(1, lab[idx], sim.clamp(min=0))
        p = votes / votes.sum(1, keepdim=True).clamp(min=1e-12)
        out.append(torch.log(p + FLOOR).cpu().numpy())
    return np.vstack(out)


def knn_tune(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    n_classes: int,
    ks: tuple[int, ...] = (1, 3, 5, 10, 20, 50),
) -> int:
    """k with the best macro recall on validation (ties: smaller k)."""
    best, best_k = -1.0, ks[0]
    for k in [k for k in ks if k <= len(xtr)] or [1]:
        pred = knn_logits(xtr, ytr, xva, k, n_classes).argmax(1)
        score = macro_recall(yva, pred)
        if score > best + 1e-9:
            best, best_k = score, k
    return best_k


def macro_recall(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


# ---------------------------------------------------------------- linear / MLP


@dataclass
class TrainCfg:
    hidden: int = 0  # 0 = multinomial logistic regression
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 512
    min_steps: int = 50  # per evaluation round, so small training sets still get trained
    max_rounds: int = 40
    patience: int = 3
    seed: int = 20261027


def _net(d: int, c: int, cfg: TrainCfg) -> nn.Module:
    if cfg.hidden == 0:
        return nn.Linear(d, c)
    return nn.Sequential(
        nn.Linear(d, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, c)
    )


def train_net(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    n_classes: int,
    cfg: TrainCfg,
) -> nn.Module:
    torch.manual_seed(cfg.seed)
    freq = np.bincount(ytr, minlength=n_classes).astype(float)
    cw = np.where(freq > 0, 1 / np.sqrt(np.maximum(freq, 1)), 0.0)
    cw = cw / (cw * freq).sum() * freq.sum()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE))
    net = _net(xtr.shape[1], n_classes, cfg).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt, yt = torch.from_numpy(xtr).to(DEVICE), torch.from_numpy(ytr).to(DEVICE)
    xv, yv = torch.from_numpy(xva).to(DEVICE), torch.from_numpy(yva).to(DEVICE)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    batch = min(cfg.batch, max(16, len(xt) // 4))
    best, best_state, bad = float("inf"), None, 0
    for _ in range(cfg.max_rounds):  # a round = max(one pass, min_steps steps)
        net.train()
        steps = 0
        while steps < cfg.min_steps or steps * batch < len(xt):
            for idx in torch.randperm(len(xt), generator=g).split(batch):
                idx = idx.to(DEVICE)
                opt.zero_grad()
                loss_fn(net(xt[idx]), yt[idx]).backward()
                opt.step()
                steps += 1
                if steps >= cfg.min_steps and steps * batch >= len(xt):
                    break
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(xv), yv))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    return net.eval()


def net_logits(net: nn.Module, x: np.ndarray, chunk: int = 8192) -> np.ndarray:
    with torch.no_grad():
        return np.vstack(
            [
                net(torch.from_numpy(x[s : s + chunk]).to(DEVICE)).cpu().numpy()
                for s in range(0, len(x), chunk)
            ]
        )


# ---------------------------------------------------------------- calibration


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Scalar T minimising validation NLL of softmax(logits / T)."""
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top-1 probability."""
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)
