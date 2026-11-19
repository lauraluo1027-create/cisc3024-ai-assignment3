"""Shared training routine for RPCANet-lite (used by train.py and ablation.py)."""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from models import RPCANetLite

GAMMA = 4.0
_tr = np.load("../data/train.npz"); _te = np.load("../data/test.npz")
Xtr = torch.tensor(_tr["X"], dtype=torch.float32); Ytr = torch.tensor(_tr["Y"], dtype=torch.float32)
Xte = torch.tensor(_te["X"], dtype=torch.float32); Yte = _te["Y"]

bce = nn.BCEWithLogitsLoss(reduction="none")


def loss_fn(u, gt):
    logits = GAMMA * (u - 1.0)
    w = 1.0 + 20.0 * gt
    l_bce = (bce(logits, gt) * w).mean()
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * gt).sum() + 1) / (p.sum() + gt.sum() + 1)
    return l_bce + dice


@torch.no_grad()
def val_auc(model):
    model.eval()
    aucs = []
    for j in range(Xte.shape[0]):
        u = model.detector(Xte[j:j + 1])[0, 0].numpy()
        aucs.append(roc_auc_score((Yte[j, 0] > 0.5).ravel(), u.ravel()))
    model.train()
    return float(np.mean(aucs))


def train_model(stages=5, epochs=25, use_head=True, seed=0, lr=5e-2, bs=4, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RPCANetLite(stages=stages, use_head=use_head)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    hist = {"loss": [], "val_auc": [], "epoch_seconds": []}
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n); tot = nb = 0
        t_ep = time.time()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            u = model.detector(Xtr[idx])
            loss = loss_fn(u, (Ytr[idx] > 0.5).float())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        auc = val_auc(model)
        hist["loss"].append(tot / nb); hist["val_auc"].append(auc)
        hist["epoch_seconds"].append(time.time() - t_ep)
        if verbose:
            print(f"  epoch {ep+1:02d}/{epochs}  loss={tot/nb:.4f}  val_AUC={auc:.4f}  ({time.time()-t_ep:.1f}s)", flush=True)
    model.eval()
    return model, hist, time.time() - t0, sum(p.numel() for p in model.parameters())


def learned_params(model):
    out = {}
    for i, st in enumerate(model.stages):
        out[f"stage{i+1}"] = {"rho": float(st.rho), "a": float(st.a), "lam": float(st.lam)}
    return out
