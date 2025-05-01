# src/evaluate.py
"""
Evaluate a trained model on test data and save a confusion-matrix PNG.

Usage:
    python -m src.evaluate --weights checkpoints/best.pt
"""
import argparse, pathlib, torch, matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score,
                             RocCurveDisplay)
from torch.utils.data import DataLoader
from src.dataset import CoronaHackDataset
from src.model import CovidResNet

@torch.inference_mode()
def evaluate(model, loader, device):
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        y_true.extend(y.numpy())
        y_prob.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    y_pred = [p > 0.5 for p in y_prob]
    return y_true, y_pred, y_prob           # for metrics

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CovidResNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()                                  # 🔑 inference mode

    test_ds = CoronaHackDataset("data/interim/test_manifest.csv")
    test_ld = DataLoader(test_ds, batch_size=32)

    y_true, y_pred, y_prob = evaluate(model, test_ld, device)

    # 1️⃣  Text metrics
    print(classification_report(y_true, y_pred, target_names=["Normal","COVID"]))

    # 2️⃣  AUROC
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUROC: {auc:.3f}")

    # 3️⃣  Confusion-matrix PNG
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal","COVID"])
    disp.plot(cmap="Blues", values_format="d")
    out = pathlib.Path("reports"); out.mkdir(exist_ok=True)
    plt.title("Confusion Matrix"); plt.savefig(out/"confusion_matrix.png")
    plt.close()

    # 4️⃣  ROC curve (optional)
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve"); plt.savefig(out/"roc_curve.png")
    print("✓ metrics & plots saved in /reports")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=pathlib.Path, required=True)
    main(p.parse_args())
