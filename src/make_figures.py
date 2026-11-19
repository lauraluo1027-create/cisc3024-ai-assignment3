"""Generate every figure used in the report (AI-written)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from models import RPCANetLite
from baselines import top_hat, pca_residual, ipi_rpca

K = 5
FIG = "../figures/"
te = np.load("../data/test.npz"); X = te["X"][:, 0]; Y = te["Y"][:, 0]

# ---------- 1. training curve ----------
log = json.load(open(f"../results/train_log_stage{K}.json"))
fig, ax1 = plt.subplots(figsize=(6.2, 3.4))
ep = np.arange(1, len(log["loss"]) + 1)
ax1.plot(ep, log["loss"], "C0-o", ms=3, label="training loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="C0")
ax2 = ax1.twinx()
ax2.plot(ep, log["val_auc"], "C3-s", ms=3, label="validation AUC")
ax2.set_ylabel("validation AUC", color="C3")
ax1.set_title(f"Training of RPCANet-lite (K={K}, {log['params_count']} parameters)")
fig.tight_layout(); fig.savefig(FIG + "fig_training_curve.png", dpi=180); plt.close(fig)

# ---------- 2. ROC curves ----------
S = json.load(open("../results/eval_summary.json"))
fig, ax = plt.subplots(figsize=(5.2, 4.6))
for nm, (fa, pd) in S["curves"].items():
    fa = np.array(fa); pd = np.array(pd)
    ax.plot(np.clip(fa, 1e-7, None), pd, label=f"{nm} (AUC={[r['auc'] for r in S['rows'] if r['method']==nm][0]:.4f})")
ax.set_xscale("log"); ax.set_xlabel("false alarm rate (log)"); ax.set_ylabel("detection rate Pd")
ax.set_title("ROC on the synthetic IR test set"); ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(FIG + "fig_roc.png", dpi=180); plt.close(fig)

# ---------- 3. qualitative comparison ----------
model = RPCANetLite(stages=K)
model.load_state_dict(torch.load(f"../results/rpcanet_stage{K}.pt")); model.eval()

def norm(m):
    return (m - m.min()) / (m.max() - m.min() + 1e-12)

idxs = [0, 7, 23]
cols = ["Input image", "Ground truth", "Top-hat", "PCA residual", "IPI (RPCA)", f"RPCANet-lite K={K}"]
fig, axes = plt.subplots(len(idxs), len(cols), figsize=(13, 2.3 * len(idxs)))
with torch.no_grad():
    for r, i in enumerate(idxs):
        x = X[i]
        ours = model.detector(torch.tensor(x)[None, None])[0, 0].numpy()
        imgs = [x, Y[i], top_hat(x), pca_residual(x), ipi_rpca(x), ours]
        for c, (img, name) in enumerate(zip(imgs, cols)):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray" if c != 1 else "hot")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(name, fontsize=9)
            if c == 1:   # mark GT contour on input otherwise
                pass
            if c == 0:
                ax.contour(Y[i] > 0.5, levels=[0.5], colors="lime", linewidths=1.2)
fig.suptitle("Qualitative comparison (green contour = ground-truth target)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG + "fig_qualitative.png", dpi=170); plt.close(fig)

# ---------- 4. ablation ----------
A = json.load(open("../results/ablation.json"))
head = [a for a in A if a["head"]]
nohead = [a for a in A if not a["head"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
ks = [a["stages"] for a in head]
a1.plot(ks, [a["auc"] for a in head], "C0-o", label="with sparsity head")
a1.plot(ks, [a["iou"] for a in head], "C2-s", label="IoU (best threshold)")
if nohead:
    a1.axhline(nohead[0]["auc"], ls="--", c="C3", label=f"no head (K={nohead[0]['stages']}) AUC")
a1.set_xlabel("number of unrolled stages K"); a1.set_ylabel("score"); a1.grid(alpha=.3)
a1.set_title("Effect of unrolled stages"); a1.legend(fontsize=8)
a2.plot(ks, [a["infer_ms"] for a in head], "C4-^")
a2.set_xlabel("number of unrolled stages K"); a2.set_ylabel("inference time (ms / 128x128 image)")
a2.set_title("Computational cost"); a2.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG + "fig_ablation.png", dpi=180); plt.close(fig)

# ---------- 5. learned parameters ----------
lp = log["learned"]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
st = list(lp.keys()); w = 0.27
for j, (key, color) in enumerate(zip(["rho", "a", "lam"], ["C0", "C1", "C2"])):
    ax.bar(np.arange(len(st)) + (j - 1) * w, [lp[s][key] for s in st], w, label=key, color=color)
ax.set_xticks(np.arange(len(st))); ax.set_xticklabels([s.replace("stage", "stage ") for s in st])
ax.set_ylabel("learned value"); ax.legend(fontsize=8)
ax.set_title("Learned unfolding parameters per stage")
fig.tight_layout(); fig.savefig(FIG + "fig_params.png", dpi=180); plt.close(fig)
print("figures:", ["fig_training_curve.png", "fig_roc.png", "fig_qualitative.png", "fig_ablation.png", "fig_params.png"])
