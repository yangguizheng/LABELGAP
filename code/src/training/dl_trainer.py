"""Unified training loop for the DL baselines (protocol section 8).

Identical for every DL model: AdamW, unweighted cross-entropy, same batch size,
same early-stopping rule on validation loss. The test set is only touched
after training, with the best (validation-selected) checkpoint."""
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
from src.utils import set_seed


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def median_time(fn, repeats: int = 3) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def train_dl(name: str, model_cfg: dict, train_cfg: dict, data: dict, seed: int,
             ckpt_dir: Path, log: logging.Logger, device: str = "cuda",
             class_weights: np.ndarray | None = None) -> dict:
    """class_weights (D3 diagnostic only) weight both the training loss and the
    validation loss used for early stopping; None reproduces Stage I exactly."""
    set_seed(seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(data["class_names"])
    in_dim = data["X_train"].shape[1]
    model = build_model(name, in_dim, n_classes, model_cfg["params"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(t.numel() * t.element_size()
                     for t in list(model.parameters()) + list(model.buffers()))
    flops = count_flops(model, in_dim, device)
    log.info("%s seed=%d params=%d flops/sample=%d", name, seed, n_params, flops)

    x_tr = torch.from_numpy(data["X_train"]).to(device)
    y_tr = torch.from_numpy(data["y_train"]).long().to(device)
    x_va = torch.from_numpy(data["X_val"]).to(device)
    y_va = torch.from_numpy(data["y_val"]).long().to(device)
    y_va_np = data["y_val"]
    # A-D1 monitoring only (never used for selection): affected validation rows, clean labels
    mon = ("y_val_clean" in data and "y_val_noisy" in data)
    if mon:
        aff_va = data["y_val_clean"] != data["y_val_noisy"]
        y_va_clean_aff = data["y_val_clean"][aff_va]
    fixed = train_cfg.get("fixed_epochs")

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"])
    cw = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32,
                                                            device=device)
    if train_cfg.get("loss_name", "ce") == "gce":  # A-D1 diagnostic only
        from src.training.noise_robust import GCELoss
        loss_fn = GCELoss(train_cfg["gce_q"])
    else:
        loss_fn = nn.CrossEntropyLoss(weight=cw)
    bs, eval_bs = train_cfg["batch_size"], train_cfg["eval_batch_size"]
    es = train_cfg["early_stopping"]
    clip = train_cfg.get("grad_clip_norm")
    gen = torch.Generator().manual_seed(seed)
    n = len(x_tr)
    steps = n // bs  # drop_last: avoids a size-1 BatchNorm batch

    history, best_loss, best_epoch, bad = [], float("inf"), 0, 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_time, t_start = 0.0, time.perf_counter()
    for epoch in range(1, (fixed or train_cfg["max_epochs"]) + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=gen).to(device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            xb, yb = x_tr.index_select(0, idx), y_tr.index_select(0, idx)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            loss_sum += loss.detach() * len(idx)
            correct += (logits.argmax(1) == yb).sum()
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time

        val_logits = predict_logits(model, x_va, eval_bs)
        val_loss = loss_fn(val_logits, y_va).item()  # same criterion as training
        val_pred = val_logits.argmax(1).cpu().numpy()
        val_f1 = f1_score(y_va_np, val_pred, average="macro", labels=np.arange(n_classes),
                          zero_division=0)
        improved = val_loss < best_loss - es["min_delta"]
        if improved:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "val_macro_f1": val_f1}, ckpt_dir / "best.pt")
        else:
            bad += 1
        row = {"epoch": epoch, "train_loss": (loss_sum / (steps * bs)).item(),
               "train_acc": correct.item() / (steps * bs), "val_loss": val_loss,
               "val_acc": float((val_pred == y_va_np).mean()), "val_macro_f1": val_f1,
               "epoch_train_time_s": epoch_time, "is_best": improved}
        if mon:
            row["val_recovery_clean"] = float((val_pred[aff_va] == y_va_clean_aff).mean())
        history.append(row)
        log.info("%s seed=%d ep=%03d train_loss=%.4f val_loss=%.4f val_acc=%.4f "
                 "val_macroF1=%.4f %.1fs%s", name, seed, epoch, row["train_loss"], val_loss,
                 row["val_acc"], val_f1, epoch_time, " *" if improved else "")
        if bad >= es["patience"] and not fixed:
            break
    wall_time = time.perf_counter() - t_start
    if fixed:  # fixed budget: the evaluated model is the last epoch
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                    "val_macro_f1": val_f1, "selection": "last epoch (fixed budget)"}, ckpt_dir / "best.pt")
    peak_mem_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "epoch": epoch}, ckpt_dir / "last.pt")

    # "last" checkpoint predictions are kept only as a transparency check
    x_te_np = data["X_test"]
    proba_test_last = F.softmax(predict_logits(model, torch.from_numpy(x_te_np).to(device),
                                               eval_bs), 1).cpu().numpy()

    state = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model_state"])
    proba_val = F.softmax(predict_logits(model, x_va, eval_bs), 1).cpu().numpy()

    def infer():  # end-to-end: host numpy array -> host probabilities
        x = torch.from_numpy(x_te_np).to(device)
        return F.softmax(predict_logits(model, x, eval_bs), 1).cpu().numpy()

    proba_test = infer()
    infer_s = median_time(infer)

    efficiency = {
        "params": int(n_params),
        "flops_per_sample": int(flops),
        "model_size_mb": size_bytes / 2 ** 20,
        "train_time_s": train_time,
        "train_wall_time_incl_val_s": wall_time,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "time_per_epoch_s": train_time / epoch,
        "test_inference_time_s": infer_s,
        "test_inference_us_per_sample": infer_s / len(x_te_np) * 1e6,
        "peak_gpu_memory_mb": peak_mem_mb,
        "inference_batch_size": eval_bs,
    }
    return {"proba_val": proba_val, "proba_test": proba_test,
            "proba_test_last": proba_test_last, "history": pd.DataFrame(history),
            "efficiency": efficiency}
