"""
Train a COVID-19 chest-X-ray classifier.

Default config: src/config.yaml
CLI examples
------------
python -m src.train                    # normal
python -m src.train --epochs 3         # quick test
python -m src.train --config custom.yaml
"""
import argparse, pathlib, yaml, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset import CoronaHackDataset
from src.model   import CovidResNet


# ── helpers ──────────────────────────────────────────────────────────
def load_cfg(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def make_loaders(cfg):
    bs, nw = cfg["train"]["batch_size"], cfg["data"]["num_workers"]
    train_ds = CoronaHackDataset(cfg["data"]["train_manifest"], augment=True)
    val_ds   = CoronaHackDataset(cfg["data"]["val_manifest"])
    return (DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw),
            DataLoader(val_ds,   batch_size=bs,                num_workers=nw))


def make_loss(cfg, device):
    weights = torch.tensor(cfg["train"]["class_weights"]).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def make_scheduler(opt, cfg):
    if cfg["train"]["scheduler"] == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    if cfg["train"]["scheduler"] == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
    return None


def train_epoch(model, loader, loss_fn, opt, device):
    model.train(); total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.inference_mode()
def val_acc(model, loader, device):
    model.eval(); hit = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        hit += (model(x).argmax(1) == y).sum().item()
        tot += y.size(0)
    return hit / tot


# ── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_ld, va_ld = make_loaders(cfg)

    model   = CovidResNet().to(device)
    loss_fn = make_loss(cfg, device)
    opt     = optim.AdamW(model.parameters(),
                          lr=cfg["train"]["lr"],
                          weight_decay=cfg["train"]["weight_decay"])
    sched   = make_scheduler(opt, cfg)

    ckpt_dir = pathlib.Path(cfg["logging"]["save_dir"]); ckpt_dir.mkdir(exist_ok=True)
    writer   = SummaryWriter(cfg["logging"]["tb_logdir"]) if cfg["logging"]["tensorboard"] else None

    best, patience_left = 0.0, cfg["train"]["early_stopping_patience"]

    for ep in range(1, cfg["train"]["epochs"] + 1):
        tr_loss = train_epoch(model, tr_ld, loss_fn, opt, device)
        acc     = val_acc(model, va_ld, device)

        if writer:
            writer.add_scalar("Loss/train", tr_loss, ep)
            writer.add_scalar("Acc/val",    acc,     ep)

        if sched and cfg["train"]["scheduler"] == "reduce_on_plateau":
            sched.step(acc)
        elif sched:
            sched.step()

        print(f"Epoch {ep}/{cfg['train']['epochs']}  "
              f"train_loss={tr_loss:.4f}  val_acc={acc:.3f}")

        if acc > best:
            best, patience_left = acc, cfg["train"]["early_stopping_patience"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("⚑ Early stopping.")
                break

    if writer: writer.close()
    print(f"✓ best val-acc {best:.3f}  →  {ckpt_dir/'best.pt'}")


if __name__ == "__main__":
    main()
