"""
Minimal training loop:
python -m src.train --epochs 10
"""
import argparse, pathlib, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import CoronaHackDataset
from src.model import CovidResNet

def train_one_epoch(model, loader, loss_fn, optimiser, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimiser.step()

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = CoronaHackDataset("data/interim/train_manifest.csv", augment=True)
    val_ds   = CoronaHackDataset("data/interim/val_manifest.csv")
    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_ld   = DataLoader(val_ds,   batch_size=32)

    model = CovidResNet().to(device)
    loss_fn  = nn.CrossEntropyLoss()
    optimser = optim.AdamW(model.parameters(), lr=1e-4)

    best_acc = 0.0
    ckpt_dir = pathlib.Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs+1):
        train_one_epoch(model, train_ld, loss_fn, optimser, device)
        acc = evaluate(model, val_ld, device)
        print(f"epoch {epoch}/{args.epochs}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_dir/"best.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    main(ap.parse_args())
