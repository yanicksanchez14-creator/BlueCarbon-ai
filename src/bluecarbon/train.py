"""Training loop: CE + Dice, class weighting, AMP, cosine LR, early stopping on val mIoU."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import ChipDataset, class_frequencies, fit_normalizer
from .metrics import confusion, summarize
from .model import build_model, resolve_device, save_checkpoint
from .schema import IGNORE_INDEX, N_CLASSES
from .tiling import ChipRecord


class DiceCELoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
        self.dw = dice_weight

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, y)
        if self.dw <= 0:
            return loss
        valid = (y != IGNORE_INDEX).unsqueeze(1)
        prob = logits.float().softmax(1) * valid
        onehot = F.one_hot(y.clamp(max=N_CLASSES - 1), N_CLASSES).permute(0, 3, 1, 2) * valid
        inter = (prob * onehot).sum((0, 2, 3))
        den = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
        present = onehot.sum((0, 2, 3)) > 0
        dice = 1 - (2 * inter + 1) / (den + 1)
        return loss + self.dw * (dice[present].mean() if present.any() else 0.0)


def class_weights(freq: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    f = np.maximum(freq, 1).astype(np.float64)
    w = 1 / f if mode == "inverse" else 1 / np.sqrt(f)
    w = w / w[freq > 0].mean()
    w[freq == 0] = 0.0
    return np.clip(w, 0, 10).astype(np.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        cm += confusion(y.numpy(), pred)
    return cm


def train(cfg: Config, records: list[ChipRecord], out_dir: str | Path, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = [r for r in records if r.split == "train"]
    va = [r for r in records if r.split == "val"]
    te = [r for r in records if r.split == "test"]
    if not tr or not va:
        raise ValueError(f"Need train and val chips (got train={len(tr)}, val={len(va)})")
    log(f"chips: train={len(tr)} val={len(va)} test={len(te)}")

    torch.manual_seed(cfg.chips.seed)
    np.random.seed(cfg.chips.seed)
    device = resolve_device(cfg.train.device)
    norm = fit_normalizer(tr)
    freq = class_frequencies(tr)
    w = class_weights(freq, cfg.train.class_weighting)
    log(f"train class pixels: {freq.tolist()}  weights: {None if w is None else np.round(w, 2).tolist()}")

    t = cfg.train
    kw = dict(batch_size=t.batch_size, num_workers=t.num_workers, pin_memory=device.type == "cuda")
    dl_tr = DataLoader(ChipDataset(tr, norm, augment=True), shuffle=True, drop_last=len(tr) > t.batch_size, **kw)
    dl_va = DataLoader(ChipDataset(va, norm), shuffle=False, **kw)

    m = cfg.model
    model = build_model(m.arch, m.encoder, m.encoder_weights).to(device)
    lossf = DiceCELoss(None if w is None else torch.tensor(w, device=device), t.dice_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t.lr, total_steps=max(1, t.epochs * len(dl_tr)),
                                                pct_start=0.1)
    use_amp = t.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_epoch, history = -1.0, -1, []
    ckpt = out / "model.pt"
    for epoch in range(1, t.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        val = summarize(evaluate(model, dl_va, device))
        history.append({"epoch": epoch, "loss": tot / max(nb, 1), **{k: val[k] for k in ("mIoU", "macro_f1",
                                                                                           "overall_accuracy")}})
        log(f"epoch {epoch:3d} loss {tot / max(nb, 1):.4f} val mIoU {val['mIoU']:.4f} "
            f"F1 {val['macro_f1']:.4f} OA {val['overall_accuracy']:.4f} ({time.time() - t0:.0f}s)")
        if val["mIoU"] > best:
            best, best_epoch = val["mIoU"], epoch
            save_checkpoint(ckpt, model, m.arch, m.encoder, norm, {"val": val, "epoch": epoch})
        elif epoch - best_epoch >= t.patience:
            log(f"early stop: no val improvement for {t.patience} epochs")
            break

    from .model import load_checkpoint

    model, norm, ck = load_checkpoint(ckpt, device)
    results = {"best_epoch": best_epoch, "val": ck["metrics"]["val"], "history": history}
    if te:
        cm_te = evaluate(model, DataLoader(ChipDataset(te, norm), shuffle=False, **kw), device)
        results["test"] = summarize(cm_te)
        results["test_confusion"] = cm_te.tolist()
        log(f"TEST mIoU {results['test']['mIoU']:.4f}  macro-F1 {results['test']['macro_f1']:.4f}  "
            f"OA {results['test']['overall_accuracy']:.4f}")
    per_site = {}
    for site in sorted({r.site for r in te}):
        rs = [r for r in te if r.site == site]
        per_site[site] = summarize(evaluate(model, DataLoader(ChipDataset(rs, norm), shuffle=False, **kw), device))
    results["test_per_site"] = per_site
    save_checkpoint(ckpt, model, m.arch, m.encoder, norm, {**ck["metrics"], "test": results.get("test"),
                                                           "test_confusion": results.get("test_confusion")})
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    return results
