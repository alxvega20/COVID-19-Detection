"""
Train a COVID-19 chest-X-ray classifier.

# default run (uses src/config.yaml)
python -m src.train

# quick test run (override epochs)
python -m src.train --epochs 5
"""
import argparse, pathlib, yaml, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import CoronaHackDataset
from src.model import CovidResNet


# ───────────────────────── helpers ──────────────────────────
def load_cfg(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def make_loaders(cfg):
    tr_ds = CoronaHackDataset(cfg["data"]["train_manifest"], augment=True)
    va_ds = CoronaHackDataset(cfg["data"]["val_manifest"])
    kw = dict(num_workers=cfg["data"]["num_workers"],
              batch_size=cfg["train"]["batch_size"])
    return (DataLoader(tr_ds, shuffle=True,  **kw),
            DataLoader(va_ds,                 **kw))


def make_loss(cfg, device):
    w = torch.tensor(cfg["train"]["class_weights"]).to(device)
    return nn.CrossEntropyLoss(weight=w)


def make_scheduler(opt, cfg):
    if cfg["train"]["scheduler"] == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt,
                                                    T_max=cfg["train"]["epochs"])
    if cfg["train"]["scheduler"] == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
    return None


# ───────────────────────── train / val loops ──────────────────────────
def train_epoch(model, loader, loss_fn, opt, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()


@torch.inference_mode()
def val_acc(model, loader, device):
    model.eval()
    hit = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        hit += (model(x).argmax(1) == y).sum().item()
        tot += y.size(0)
    return hit / tot


# ─────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--epochs", type=int, help="override epochs")
    cfg = load_cfg(ap.parse_args().config)
    if ap.parse_args().epochs:
        cfg["train"]["epochs"] = ap.parse_args().epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_ld, va_ld = make_loaders(cfg)

    model = CovidResNet().to(device)
    loss_fn = make_loss(cfg, device)
    opt = optim.AdamW(model.parameters(),
                      lr=cfg["train"]["lr"],
                      weight_decay=cfg["train"]["weight_decay"])
    sched = make_scheduler(opt, cfg)

    ckpt_dir = pathlib.Path(cfg["logging"]["save_dir"]); ckpt_dir.mkdir(exist_ok=True)
    best, patience = 0.0, cfg["train"]["early_stopping_patience"]

    for ep in range(1, cfg["train"]["epochs"] + 1):
        train_epoch(model, tr_ld, loss_fn, opt, device)
        acc = val_acc(model, va_ld, device)
        if sched and cfg["train"]["scheduler"] == "reduce_on_plateau":
            sched.step(acc)
        elif sched:
            sched.step()

        print(f"epoch {ep}/{cfg['train']['epochs']}  val_acc={acc:.3f}")

        if acc > best:
            best, patience_left = acc, patience  # reset patience
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
