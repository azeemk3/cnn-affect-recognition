import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from utils.metrics import basic_cls_metrics, rmse, ccc, sagr

def _to_device(batch, device, non_blocking=True):
    imgs, y_cls, y_reg, reg_ok, paths = batch
    return (imgs.to(device, non_blocking=non_blocking),
            y_cls.to(device, non_blocking=non_blocking),
            y_reg.to(device, non_blocking=non_blocking),
            reg_ok.to(device, non_blocking=non_blocking),
            paths)

def train_one_epoch(model, loader, optimizer, device, lambda_reg=1.0, scaler=None):
    model.train()
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    tot_loss = tot_cls = tot_reg = 0.0

    for batch in tqdm(loader, leave=False):
        imgs, y_cls, y_reg, reg_ok, _ = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                            dtype=torch.float16 if device.type=="cuda" else torch.bfloat16):
            logits, reg = model(imgs)
            loss_cls = ce(logits, y_cls)
            mask = reg_ok.bool()
            loss_reg = mse(reg[mask], y_reg[mask]) if mask.any() else torch.tensor(0.0, device=device)
            loss = loss_cls + lambda_reg * loss_reg

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()

        bs = imgs.size(0)
        tot_loss += loss.item()*bs
        tot_cls  += loss_cls.item()*bs
        tot_reg  += (loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg)*bs

    n = len(loader.dataset)
    return {"loss": tot_loss/n, "cls": tot_cls/n, "reg": tot_reg/n}

@torch.no_grad()
def evaluate(model, loader, device, lambda_reg=1.0):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    mse = nn.MSELoss(reduction="sum")
    total_cls = total_reg = 0.0
    y_true_cls, y_pred_logits = [], []
    y_true_reg, y_pred_reg, y_mask = [], [], []

    for batch in tqdm(loader, leave=False):
        imgs, y_cls, y_reg, reg_ok, _ = _to_device(batch, device)
        with torch.autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                            dtype=torch.float16 if device.type=="cuda" else torch.bfloat16):
            logits, reg = model(imgs)

        total_cls += ce(logits, y_cls).item()
        mask = reg_ok.bool()
        if mask.any(): total_reg += mse(reg[mask], y_reg[mask]).item()

        y_true_cls.append(y_cls.cpu().numpy())
        y_pred_logits.append(logits.float().cpu().numpy())
        y_true_reg.append(y_reg.cpu().numpy())
        y_pred_reg.append(reg.float().cpu().numpy())
        y_mask.append(mask.cpu().numpy())

    y_true_cls = np.concatenate(y_true_cls)
    y_pred_logits = np.concatenate(y_pred_logits)
    y_true_reg = np.concatenate(y_true_reg)
    y_pred_reg = np.concatenate(y_pred_reg)
    y_mask = np.concatenate(y_mask).astype(bool)

    metrics = basic_cls_metrics(y_true_cls, y_pred_logits)
    if y_mask.sum() > 0:
        t, p = y_true_reg[y_mask], y_pred_reg[y_mask]
        metrics.update({"rmse": rmse(t, p), "ccc": ccc(t, p), "sagr": sagr(t, p)})
    else:
        metrics.update({"rmse": 0.0, "ccc": 0.0, "sagr": 0.0})
    return metrics
