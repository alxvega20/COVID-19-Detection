"""
Predict the class of a single chest-X-ray.

python -m src.predict --img sample.jpg --weights checkpoints/best.pt
"""
import argparse, torch
from PIL import Image
import torchvision.transforms as T
from src.model import CovidResNet


def load_img(path):
    tf = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)  # 1×C×H×W


@torch.inference_mode()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CovidResNet().to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device))

    prob = torch.softmax(model(load_img(args.img).to(device)), 1)[0,1].item()
    label = "COVID-19" if prob > 0.5 else "Normal"
    print(f"{args.img}: {label} (p_covid={prob:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img",     required=True)
    ap.add_argument("--weights", required=True)
    main(ap.parse_args())
