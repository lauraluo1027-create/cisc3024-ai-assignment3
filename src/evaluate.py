"""Evaluate all methods on the held-out synthetic test set (AI-written).

Metrics
  AUC      : pixel-level ROC AUC (threshold-free, primary metric)
  IoU*     : best pixel-level IoU over a global threshold grid (upper bound)
  Pd@Fa1e-4: detection rate at false-alarm rate 1e-4
  time     : seconds per 128x128 image (CPU)
"""
import json, time, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from models import RPCANetLite
from baselines import top_hat, pca_residual, ipi_rpca

STAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 5
te = np.load("../data/test.npz"); X = te["X"][:, 0]; Y = te["Y"][:, 0]

# ---------- 1) run every detector ----------
methods = {}
t = time.time(); methods["Top-hat"] = np.stack([top_hat(x) for x in X]); 
methods["Top-hat_time"] = (time.time() - t) / len(X)
t = time.time(); methods["PCA-residual (rank-3)"] = np.stack([pca_residual(x) for x in X])
methods["PCA_time"] = (time.time() - t) / len(X)
t = time.time(); methods["IPI (RPCA/IALM)"] = np.stack([ipi_rpca(x) for x in X])
methods["IPI_time"] = (time.time() - t) / len(X)

model = RPCANetLite(stages=STAGES)
model.load_state_dict(torch.load(f"../results/rpcanet_stage{STAGES}.pt"))
model.eval()
t = time.time()
with torch.no_grad():
    ours = np.concatenate([model.detector(torch.tensor(X[i:i + 8])).numpy() for i in range(0, len(X), 8)])[:, 0]
methods[f"RPCANet-lite (K={STAGES})"] = ours
methods["RPCANet_time"] = (time.time() - t) / len(X)

names = [k for k in methods if not k.endswith("_time")]
times = {"Top-hat": methods["Top-hat_time"],
         "PCA-residual (rank-3)": methods["PCA_time"],
         "IPI (RPCA/IALM)": methods["IPI_time"],
         f"RPCANet-lite (K={STAGES})": methods["RPCANet_time"]}

# ---------- 2) metrics ----------
def norm(m):
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo + 1e-12)

def roc_curve(maps, n=60):
    ths = np.linspace(0.02, 0.98, n)
    pds, fas = [], []
    gts = [(y > 0.5) for y in Y]
    ng = sum(g.sum() for g in gts)
    for th in ths:
        tp = fp = 0
        for m, g in zip(maps, gts):
            p = norm(m) > th
            tp += np.logical_and(p, g).sum(); fp += np.logical_and(p, ~g).sum()
        pds.append(tp / ng); fas.append(fp / (len(maps) * gts[0].size))
    return np.array(fas), np.array(pds), ths

from scipy.ndimage import binary_dilation

def best_iou(maps, tol=0):
    gts = [(y > 0.5) for y in Y]
    best = 0.0
    for th in np.linspace(0.05, 0.95, 19):
        ious = []
        for m, g in zip(maps, gts):
            p = norm(m) > th
            if tol:   # 1-pixel tolerance: a hit within 1 px counts as correct
                p = binary_dilation(p, iterations=tol); g = binary_dilation(g, iterations=tol)
            tp = np.logical_and(p, g).sum(); fp = np.logical_and(p, ~g).sum(); fn = np.logical_and(~p, g).sum()
            ious.append(tp / (tp + fp + fn + 1e-12))
        best = max(best, float(np.mean(ious)))
    return best

rows, curves = [], {}
for nm in names:
    maps = methods[nm]
    aucs = [roc_auc_score((y > 0.5).ravel(), m.ravel()) for m, y in zip(maps, Y)]
    fa, pd, _ = roc_curve(maps)
    # Pd at Fa = 1e-4 (interp, monotone)
    idx = int(np.argmin(np.abs(fa - 1e-4)))
    rows.append(dict(method=nm, auc=float(np.mean(aucs)), iou_best=best_iou(maps),
                     iou_tol1=best_iou(maps, tol=1),
                     pd_at_fa1e4=float(pd[idx]), fa_at_point=float(fa[idx]),
                     time_s=float(times.get(nm, 0.0))))
    curves[nm] = (fa.tolist(), pd.tolist())
    print(f"{nm:26} AUC={rows[-1]['auc']:.4f}  IoU*={rows[-1]['iou_best']:.4f}  IoU*@1px={rows[-1]['iou_tol1']:.4f}  "
          f"Pd@Fa=1e-4={rows[-1]['pd_at_fa1e4']:.3f}  t={rows[-1]['time_s']*1000:.1f} ms/img")

json.dump({"rows": rows, "curves": curves}, open("../results/eval_summary.json", "w"), indent=2)
import csv
with open("../results/eval_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\n-> ../results/eval_summary.{json,csv}")
