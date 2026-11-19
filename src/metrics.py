"""Detection metrics for infrared small-target detection (AI-written)."""
import numpy as np


def norm_map(m):
    m = m.astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo + 1e-12)


def threshold_std(m, k=4.0):
    """Adaptive threshold: mean + k*std (standard in IR target detection)."""
    return m > (m.mean() + k * m.std())


def threshold_quantile(m, q=0.995):
    return m > np.quantile(m, q)


def pixel_scores(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    iou = tp / (tp + fp + fn + 1e-12)
    pd = tp / (gt.sum() + 1e-12)                       # pixel-level detection rate
    fa = fp / (~gt).sum()                              # false-alarm rate
    return dict(iou=float(iou), pd=float(pd), fa=float(fa))


def evaluate_dataset(maps, gts, k=4.0):
    ms, ps, fs, ious = [], [], [], []
    for m, g in zip(maps, gts):
        m = norm_map(m)
        r = pixel_scores(threshold_std(m, k), g > 0.5)
        ious.append(r["iou"]); ps.append(r["pd"]); fs.append(r["fa"])
    return dict(iou=float(np.mean(ious)), pd=float(np.mean(ps)), fa=float(np.mean(fs) * 1e6))


def roc_points(maps, gts, n=40):
    """Global ROC over a shared threshold grid (normalised maps)."""
    ths = np.linspace(0.05, 0.95, n)
    pd, fa = [], []
    for t in ths:
        tp = fp = ng = 0.0
        for m, g in zip(maps, gts):
            m = norm_map(m); p = m > t; gg = g > 0.5
            tp += np.logical_and(p, gg).sum(); fp += np.logical_and(p, ~gg).sum(); ng += gg.sum()
        pd.append(tp / (ng + 1e-12)); fa.append(fp / (len(maps) * gts[0].size))
    return np.array(fa), np.array(pd), ths
