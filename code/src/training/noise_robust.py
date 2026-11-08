"""Off-the-shelf noise-robust training used only by diagnostic A-D1.

* GCELoss        -- Generalized Cross Entropy (Zhang & Sabuncu, NeurIPS 2018).
* train_coteaching -- Co-teaching (Han et al., NeurIPS 2018): two networks, each
  trains on the small-loss part of every mini-batch selected by the other.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

from src.models.nets import build_model, count_flops
from src.training.dl_trainer import median_time, predict_logits
from src.utils import set_seed


class GCELoss(nn.Module):
    """L_q = (1 - p_y^q) / q, averaged over the batch."""

    def __init__(self, q: float = 0.7):
        super().__init__()
        self.q = q

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, 1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
        return ((1.0 - p ** self.q) / self.q).mean()


def forget_rate_at(epoch: int, forget_rate: float, t_k: int) -> float:
    """Linear ramp as in the reference implementation: 0 at the first epoch,
    forget_rate from epoch t_k on (epochs are 1-indexed)."""
    return forget_rate * min((epoch - 1) / max(t_k - 1, 1), 1.0)


def train_coteaching(name: str, model_cfg: dict, train_cfg: dict, data: dict, seed: int,
                     ckpt_dir: Path, log: logging.Logger, forget_rate: float, t_k: int,
                     device: str = "cuda") -> dict:
    set_seed(seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_classes, in_dim = len(data["class_names"]), data["X_train"].shape[1]
    torch.manual_seed(seed)
    net1 = build_model(name, in_dim, n_classes, model_cfg["params"]).to(device)
    torch.manual_seed(seed + 1000)
    net2 = build_model(name, in_dim, n_classes, model_cfg["params"]).to(device)
    torch.manual_seed(seed)
    n_params = sum(p.numel() for p in net1.parameters())
    size_bytes = sum(t.numel() * t.element_size() for t in list(net1.parameters()) + list(net1.buffers()))
    flops = count_flops(net1, in_dim, device)
    log.info("%s co-teaching seed=%d params/net=%d forget_rate=%.2f t_k=%d", name, seed, n_params,
             forget_rate, t_k)

    x_tr = torch.from_numpy(data["X_train"]).to(device)
    y_tr = torch.from_numpy(data["y_train"]).long().to(device)
    x_va = torch.from_numpy(data["X_val"]).to(device)
    y_va = torch.from_numpy(data["y_val"]).long().to(device)
    y_va_np = data["y_val"]
    mon = ("y_val_clean" in data and "y_val_noisy" in data)   # monitoring only
    if mon:
        aff_va = data["y_val_clean"] != data["y_val_noisy"]
        y_va_clean_aff = data["y_val_clean"][aff_va]
    fixed = train_cfg.get("fixed_epochs")
    opts = [torch.optim.AdamW(n.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
            for n in (net1, net2)]
    bs, eval_bs = train_cfg["batch_size"], train_cfg["eval_batch_size"]
    es, clip = train_cfg["early_stopping"], train_cfg.get("grad_clip_norm")
    gen = torch.Generator().manual_seed(seed)
    n = len(x_tr)
    steps = n // bs

    history, best_loss, best_epoch, bad = [], float("inf"), 0, 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_time, t_start = 0.0, time.perf_counter()
    for epoch in range(1, (fixed or train_cfg["max_epochs"]) + 1):
        rate = forget_rate_at(epoch, forget_rate, t_k)
        net1.train()
        net2.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=gen).to(device)
        keep = max(1, int(round((1.0 - rate) * bs)))
        loss_sum = torch.zeros((), device=device)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            xb, yb = x_tr.index_select(0, idx), y_tr.index_select(0, idx)
            l1, l2 = net1(xb), net2(xb)
            ce1 = F.cross_entropy(l1, yb, reduction="none").detach()
            ce2 = F.cross_entropy(l2, yb, reduction="none").detach()
            small1 = torch.argsort(ce1, stable=True)[:keep]   # chosen by net1 -> trains net2
            small2 = torch.argsort(ce2, stable=True)[:keep]   # chosen by net2 -> trains net1
            losses = (F.cross_entropy(l1[small2], yb[small2]), F.cross_entropy(l2[small1], yb[small1]))
            for net, opt, loss in zip((net1, net2), opts, losses):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(net.parameters(), clip)
                opt.step()
            loss_sum += losses[0].detach()
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time

        val_logits = predict_logits(net1, x_va, eval_bs)
        val_loss = F.cross_entropy(val_logits, y_va).item()
        val_pred = val_logits.argmax(1).cpu().numpy()
        val_f1 = f1_score(y_va_np, val_pred, average="macro", labels=np.arange(n_classes), zero_division=0)
        improved = val_loss < best_loss - es["min_delta"]
        if improved:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            torch.save({"model_state": net1.state_dict(), "net2_state": net2.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "val_macro_f1": val_f1}, ckpt_dir / "best.pt")
        else:
            bad += 1
        history.append({"epoch": epoch, "forget_rate": rate, "train_loss_net1": (loss_sum / steps).item(),
                        "val_loss": val_loss, "val_acc": float((val_pred == y_va_np).mean()),
                        "val_macro_f1": val_f1, "epoch_train_time_s": epoch_time, "is_best": improved,
                        **({"val_recovery_clean": float((val_pred[aff_va] == y_va_clean_aff).mean())} if mon else {})})
        log.info("%s co-teaching seed=%d ep=%03d rate=%.3f train_loss=%.4f val_loss=%.4f val_macroF1=%.4f %.1fs%s",
                 name, seed, epoch, rate, history[-1]["train_loss_net1"], val_loss, val_f1, epoch_time,
                 " *" if improved else "")
        if bad >= es["patience"] and not fixed:
            break
    wall_time = time.perf_counter() - t_start
    if fixed:  # fixed budget: evaluate the last epoch of network 1
        torch.save({"model_state": net1.state_dict(), "net2_state": net2.state_dict(), "epoch": epoch,
                    "selection": "last epoch (fixed budget)"}, ckpt_dir / "best.pt")
    peak_mem_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.save({"model_state": net1.state_dict(), "net2_state": net2.state_dict(), "epoch": epoch},
               ckpt_dir / "last.pt")

    x_te_np = data["X_test"]
    proba_test_last = F.softmax(predict_logits(net1, torch.from_numpy(x_te_np).to(device), eval_bs), 1).cpu().numpy()
    net1.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device)["model_state"])
    proba_val = F.softmax(predict_logits(net1, x_va, eval_bs), 1).cpu().numpy()

    def infer():
        return F.softmax(predict_logits(net1, torch.from_numpy(x_te_np).to(device), eval_bs), 1).cpu().numpy()

    proba_test = infer()
    infer_s = median_time(infer)
    efficiency = {"params": int(n_params), "flops_per_sample": int(flops), "model_size_mb": size_bytes / 2 ** 20,
                  "train_time_s": train_time, "train_wall_time_incl_val_s": wall_time, "epochs_run": epoch,
                  "best_epoch": best_epoch, "time_per_epoch_s": train_time / epoch,
                  "test_inference_time_s": infer_s, "test_inference_us_per_sample": infer_s / len(x_te_np) * 1e6,
                  "peak_gpu_memory_mb": peak_mem_mb, "inference_batch_size": eval_bs,
                  "note": "two networks trained; efficiency and predictions refer to network 1"}
    return {"proba_val": proba_val, "proba_test": proba_test, "proba_test_last": proba_test_last,
            "history": pd.DataFrame(history), "efficiency": efficiency}
