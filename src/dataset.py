import pandas as pd, torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CoronaHackDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        df = pd.read_csv(csv_path)
        self.paths   = df.filepath.values
        self.targets = df.target.values.astype("int64")

        base = [T.Resize(224), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])]
        self.transforms = T.Compose(
            [T.RandomHorizontalFlip()] + base if augment else base)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transforms(img), self.targets[idx]
