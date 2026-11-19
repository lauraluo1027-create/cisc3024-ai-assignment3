"""Ablation study (AI-written): effect of the number of unrolled stages K and
of the self-calibrated sparsity head on detection performance."""
import json, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from trainlib import train_model, learned_params, Xte, Yte
from metrics import evaluate_dataset

EPS = 15

def test_scores(model):
    maps = []
    with torch.no_grad():
        for j in range(Xte.shape[0]):
            maps.append(model.detector(Xte[j:j + 1])[0, 0].numpy())
    aucs = [roc_auc_score((Yte[j, 0] > 0.5).ravel(), maps[j].ravel()) for j in range(len(maps))]
    m = evaluate_dataset(maps, Yte[:, 0], k=4.0)
    t = time.time()
    with torch.no_grad():
        model.detector(Xte[:20])
    ms = (time.time() - t) / 20 * 1000
    return float(np.mean(aucs)), m, ms

results = []
for K in (1, 2, 3, 5, 8):
    print(f"### K={K} stages (+head)", flush=True)
    model, hist, secs, npar = train_model(stages=K, epochs=EPS, use_head=True, verbose=False)
    auc, m, ms = test_scores(model)
    results.append(dict(config=f"K={K}", stages=K, head=True, auc=auc,
                        iou=m["iou"], pd=m["pd"], fa=m["fa"], params=npar,
                        train_s=secs, infer_ms=ms, learned=learned_params(model)))
    print(f"  -> AUC={auc:.4f} IoU={m['iou']:.4f} Pd={m['pd']:.3f} Fa={m['fa']:.1f}e-6 参数={npar} 训练={secs:.0f}s", flush=True)

for K in (5,):
    print(f"### K={K} stages (no head, 纯展开 RPCA)", flush=True)
    model, hist, secs, npar = train_model(stages=K, epochs=EPS, use_head=False, verbose=False)
    auc, m, ms = test_scores(model)
    results.append(dict(config=f"K={K} no-head", stages=K, head=False, auc=auc,
                        iou=m["iou"], pd=m["pd"], fa=m["fa"], params=npar,
                        train_s=secs, infer_ms=ms, learned=learned_params(model)))
    print(f"  -> AUC={auc:.4f} IoU={m['iou']:.4f} Pd={m['pd']:.3f} 参数={npar}", flush=True)

json.dump(results, open("../results/ablation.json", "w"), indent=2)
print("\n已保存 ../results/ablation.json")
