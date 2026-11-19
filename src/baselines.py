"""Classical baselines for comparison (all written by the AI agent).

1. Top-hat   : morphological white top-hat filtering (classic IR pipeline)
2. PCA-resid : truncated-SVD (PCA) reconstruction residual on a patch matrix
3. IPI       : Infrared Patch-Image model, i.e. RPCA via inexact ALM (no learning)
"""
import numpy as np
import cv2


def top_hat(img, k=5):
    u8 = (img * 255).astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, se).astype(np.float32) / 255.0
    return th


def _patches(img, p=15, stride=15):
    h, w = img.shape
    ys = range(0, h - p + 1, stride)
    xs = range(0, w - p + 1, stride)
    idx = [(y, x) for y in ys for x in xs]
    M = np.stack([img[y:y + p, x:x + p].ravel() for y, x in idx], axis=1)
    return M, idx, p


def _unpatch(T, idx, p, h, w):
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for j, (y, x) in enumerate(idx):
        acc[y:y + p, x:x + p] += T[:, j].reshape(p, p)
        cnt[y:y + p, x:x + p] += 1
    return acc / np.maximum(cnt, 1)


def pca_residual(img, p=15, stride=15, k=3):
    """PCA (truncated SVD, rank-k) reconstruction of the patch matrix;
    the residual (original - reconstruction) is the target saliency map."""
    h, w = img.shape
    M, idx, p = _patches(img, p, stride)
    M = M - M.mean(axis=1, keepdims=True)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(k, len(s))
    T = (U[:, :k] * s[:k]) @ Vt[:k]            # rank-k PCA reconstruction
    R = M - T                                   # residual = sparse component
    return _unpatch(R, idx, p, h, w)


def ipi_rpca(img, p=15, stride=15, lam=None, iters=30):
    """IPI: RPCA (inexact ALM) on the patch matrix, no learning."""
    h, w = img.shape
    M, idx, p = _patches(img, p, stride)
    n1, n2 = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n1, n2))
    Y = M
    mu = 1.25 / np.linalg.norm(M, 2)
    rho = 1.5
    L = np.zeros_like(Y); S = np.zeros_like(Y); Yk = Y.copy()
    for _ in range(iters):
        # low-rank step: singular value soft threshold of (Yk - S + mu^-1 Yk)
        A = Y - S + Yk / mu
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s2 = np.maximum(s - 1.0 / mu, 0)
        L = (U * s2) @ Vt
        # sparse step
        B = Y - L + Yk / mu
        S = np.sign(B) * np.maximum(np.abs(B) - lam / mu, 0)
        Z = Y - L - S
        Yk = Yk + mu * Z
        mu = min(mu * rho, 1e10)
        if np.linalg.norm(Z) / (np.linalg.norm(Y) + 1e-12) < 1e-6:
            break
    return _unpatch(S, idx, p, h, w)
