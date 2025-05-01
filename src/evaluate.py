"""
Evaluate `checkpoints/best.pt` on the test split and save plots.

python -m src.evaluate --weights checkpoints/best.pt
"""
import argparse, pathlib, torch, matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, RocCurveDisplay)
from torch.utils.data import DataLoader
from src.dataset import CoronaHackDataset
from src.model   import CovidResNet


@torch.inference_mode()
def _collect(model, loader, device):
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.softmax(model(x), 1)[:, 1]   # P(COVID)
        y_true.extend(y.numpy()); y_prob.extend(prob.cpu().numpy())
    y_pred = [p > 0.5 for p in y_prob]
    return y_true, y_pred, y_prob


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CovidResNet().to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device))

    test_ds = CoronaHackDataset("data/interim/test_manifest.csv")
    test_ld = DataLoader(test_ds, batch_size=32)

    y_true, y_pred, y_prob = _collect(model, test_ld, device)

    print(classification_report(y_true, y_pred, target_names=["Normal","COVID"]))
    print(f"AUROC: {roc_auc_score(y_true, y_prob):.3f}")

    out = pathlib.Path("reports"); out.mkdir(exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Normal","COVID"]).plot(cmap="Blues")
    plt.title("Confusion Matrix"); plt.savefig(out/"confusion_matrix.png"); plt.close()

    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve"); plt.savefig(out/"roc_curve.png"); plt.close()

    print("✓ saved plots to /reports")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=pathlib.Path, required=True)
    main(p.parse_args())
