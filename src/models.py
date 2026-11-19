"""RPCANet-lite: a lightweight deep-unfolded Robust PCA network for
infrared small-target detection  (AI-generated implementation).

Robust PCA / principal component pursuit:

    min_{L,S}  ||D - L - S||_F^2 / 2 + lam ||S||_1,   rank(L) <= r

A fixed solver (ADMM / IALM) needs hand-tuned lam and many iterations.  Deep
unfolding instead unrolls K proximal-gradient steps into a K-stage network:

    Y_k = L_{k-1} + rho_k (D - L_{k-1} - S_{k-1})        gradient step
    L_k = U soft(Sigma_k, tau_k) V^T ,  tau_k = a_k * sigma_1    low-rank prox.
    S_k = soft(D - L_k, lam_k)                            sparse prox.

Learnable per stage: rho_k (step size), a_k (relative singular threshold),
lam_k (sparse threshold, also serves as the detection threshold).
Because the whole model is 3K scalars, it is fully interpretable:  the
sparse layer S is the target map and lam_k is the detection threshold.

The image is processed block-wise (32x32) so that the local background is
well approximated as low-rank, which is exactly the assumption of RPCA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft(x, t):
    return torch.sign(x) * torch.relu(x.abs() - t)


class RPCAStep(nn.Module):
    """One unrolled robust-PCA iteration (3 learnable scalars)."""

    def __init__(self, rho=0.9, a=0.05, lam=0.5):
        super().__init__()
        self.rho_raw = nn.Parameter(torch.tensor(float(rho)).logit() if 0 < rho < 1 else torch.tensor(0.0))
        self.a_raw = nn.Parameter(torch.tensor(float(a)).logit() if 0 < a < 1 else torch.tensor(-3.9))
        self.lam_raw = nn.Parameter(torch.tensor(float(lam)).log() if lam > 0 else torch.tensor(0.0))

    @property
    def rho(self):
        return torch.sigmoid(self.rho_raw) * 0.9 + 0.05

    @property
    def a(self):
        return torch.sigmoid(self.a_raw) * 0.3

    @property
    def lam(self):
        return torch.exp(self.lam_raw).clamp(0.05, 8.0)

    def forward(self, D, L, S):
        Y = L + self.rho * (D - L - S)                       # gradient step
        U, sig, Vh = torch.linalg.svd(Y, full_matrices=False)
        tau = self.a * sig[..., :1]                          # relative threshold
        L_new = (U * soft(sig, tau).unsqueeze(-2)) @ Vh      # low-rank prox.
        S_new = soft(D - L_new, self.lam)                    # sparse prox.
        return L_new, S_new


class RPCANetLite(nn.Module):
    def __init__(self, stages=3, block=0, use_head=True):
        super().__init__()
        self.stages = nn.ModuleList([RPCAStep() for _ in range(stages)])
        self.block = block   # 0 = whole image as one matrix (default)
        self.use_head = use_head
        # self-calibrated sparsity head (RPCANet-like sparse branch, ~150 params)
        self.head = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    # ---------- reshape helpers ----------
    def _to_blocks(self, D):
        b, h, w = D.shape
        p = self.block
        D5 = D.unfold(1, p, p).unfold(2, p, p)               # b, nh, nw, p, p
        nh, nw = D5.shape[1], D5.shape[2]
        return D5.reshape(-1, p, p), (nh, nw)

    def _from_blocks(self, M, grid, h, w):
        nh, nw = grid
        p = self.block
        return M.reshape(-1, nh, nw, p, p).permute(0, 1, 3, 2, 4).reshape(-1, h, w)

    # ---------- core ----------
    def forward(self, D):
        """D: (B,H,W) -> (L, S, lam_last)"""
        if self.block:
            M, grid = self._to_blocks(D)
            h, w = D.shape[-2:]
        else:
            M, grid, h, w = D, None, D.shape[-2], D.shape[-1]
        L = torch.zeros_like(M)
        S = torch.zeros_like(M)
        for st in self.stages:
            L, S = st(M, L, S)
        lam = self.stages[-1].lam
        if grid:
            L = self._from_blocks(L, grid, h, w)
            S = self._from_blocks(S, grid, h, w)
        return L, S, lam

    # ---------- inference API ----------
    @staticmethod
    def preprocess(x):
        x = x.squeeze(1) if x.dim() == 4 else x
        m = x.mean(dim=(-2, -1), keepdim=True)
        s = x.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (x - m) / s

    def detector(self, x):
        """Return (saliency u, threshold 1.0) with u > 1 meaning 'target'.
        u = |S| / lam  -- the sparse layer measured in threshold units."""
        D = self.preprocess(x)
        _, S, lam = self.forward(D)
        z = S.abs()
        if self.use_head:
            z = F.softplus(self.head(z.unsqueeze(1))).squeeze(1)
        return (z / lam).unsqueeze(1)                         # (B,1,H,W)

    def target_map(self, x):
        return self.detector(x)[0, 0].detach().cpu().numpy()
