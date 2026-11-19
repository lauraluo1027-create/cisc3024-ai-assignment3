"""Train the main RPCANet-lite model (K stages) and save weights + log."""
import json, sys
import torch
from trainlib import train_model, learned_params

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 25
print(f"训练 RPCANet-lite: K={K} stages, {EP} epochs, 训练集 300 张 128x128", flush=True)
model, hist, secs, nparam = train_model(stages=K, epochs=EP)
torch.save(model.state_dict(), f"../results/rpcanet_stage{K}.pt")
json.dump({"stages": K, "epochs": EP, "params_count": nparam,
           "loss": hist["loss"], "val_auc": hist["val_auc"],
           "epoch_seconds": hist["epoch_seconds"], "total_seconds": secs,
           "learned": learned_params(model)},
          open(f"../results/train_log_stage{K}.json", "w"), indent=2)
print(f"\n完成 {secs:.1f}s | 参数量 {nparam} | 最佳 val AUC={max(hist['val_auc']):.4f}")
print("学到的每 stage 参数:", json.dumps(learned_params(model), indent=2))
