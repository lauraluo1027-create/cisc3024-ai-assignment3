"""Synthetic infrared small-target image generator.

Simulates: (i) 1/f-like background with clouds / stripes, (ii) pixel-level
small targets (2-9 pixels) modelled as Gaussian point-spread spots.
Everything is reproducible from a seed -> fully auditable dataset.
"""
import numpy as np
import cv2

H = W = 128


def _pink_noise(rng, h=H, w=W, alpha=1.0):
    """1/f^alpha background noise in the frequency domain."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1e-6
    spec = (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))) / (f ** alpha)
    img = np.real(np.fft.ifft2(spec))
    img -= img.min()
    return img / (img.max() + 1e-12)


def _background(rng):
    img = 0.55 * _pink_noise(rng, alpha=1.1)
    # clouds: smooth large-scale blobs
    for _ in range(rng.integers(2, 5)):
        c = np.zeros((H, W), np.float32)
        cx, cy = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(c, (int(cx), int(cy)), int(rng.integers(12, 34)),
                   float(rng.uniform(0.15, 0.45)), -1)
        c = cv2.GaussianBlur(c, (0, 0), 9)
        img += c
    # directional strip noise (common in real IR sensors)
    if rng.random() < 0.5:
        img += 0.06 * np.sin(np.linspace(0, rng.uniform(6, 20), W))[None, :]
    img += 0.04 * rng.normal(size=(H, W))          # sensor noise
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img.astype(np.float32)


def _target(rng, img, mask):
    """Insert one Gaussian small target; return updated img/mask, target box."""
    sigma = rng.uniform(0.6, 1.6)                  # -> roughly 2-9 pixels
    amp = rng.uniform(0.25, 0.75)
    r = int(np.ceil(3.5 * sigma))
    cx, cy = rng.integers(r + 2, W - r - 2), rng.integers(r + 2, H - r - 2)
    yy, xx = np.mgrid[cy - r:cy + r + 1, cx - r:cx + r + 1]
    spot = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    img[cy - r:cy + r + 1, cx - r:cx + r + 1] += spot
    mask[cy - r:cy + r + 1, cx - r:cx + r + 1] = (spot > 0.35 * amp).astype(np.float32)
    return img, mask


def make_image(rng):
    img = _background(rng)
    mask = np.zeros((H, W), np.float32)
    n_tgt = int(rng.integers(1, 4))
    for _ in range(n_tgt):
        img, mask = _target(rng, img, mask)
    img = np.clip(img, 0, 1).astype(np.float32)
    return img, mask


def build(n, seed):
    imgs, masks = [], []
    rng = np.random.default_rng(seed)
    for _ in range(n):
        i, m = make_image(rng)
        imgs.append(i); masks.append(m)
    return np.stack(imgs)[:, None], np.stack(masks)[:, None]   # N,1,H,W


if __name__ == "__main__":
    Xtr, Ytr = build(300, 0)
    Xte, Yte = build(60, 12345)
    np.savez_compressed("../data/train.npz", X=Xtr, Y=Ytr)
    np.savez_compressed("../data/test.npz", X=Xte, Y=Yte)
    print("train", Xtr.shape, "test", Xte.shape,
          "targets/img:", float(Yte.sum() / Yte.shape[0]), "px avg")
    # quick visual check
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for j in range(4):
        ax[0, j].imshow(Xte[j, 0], cmap="gray"); ax[0, j].set_title("input"); ax[0, j].axis("off")
        ax[1, j].imshow(Yte[j, 0], cmap="gray"); ax[1, j].set_title("ground truth"); ax[1, j].axis("off")
    plt.tight_layout(); plt.savefig("../figures/sample_data.png", dpi=130)
    print("saved ../figures/sample_data.png")
