# src/predict.py
"""
Classify a single chest-X-ray and print probability.

Usage:
    python -m src.predict --img some_xray.jpg --weights checkpoints/best.pt
"""
import argparse, torch
from PIL import Image
import torchvision.transforms as T
from src.model import CovidResNet

def load_image(path):
    tr = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tr(Image.open(path).convert("RGB")).unsqueeze(0)   # 1×C×H×W

@torch.inference_mode()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CovidResNet().to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device))

    x = load_image(args.img).to(device)
    prob_covid = torch.softmax(model(x), 1)[0, 1].item()
    label = "COVID-19" if prob_covid > 0.5 else "Normal"
    print(f"{args.img}: {label}  (p_covid={prob_covid:.3f})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img",   type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    main(p.parse_args())
