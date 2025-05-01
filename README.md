# COVID-19-Detection

download dataset from: https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset


## Quick-start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. place the Kaggle folder here:
#    data/raw/Coronahack-Chest-XRay-Dataset

# 2. generate train/val/test CSVs
python -m src.prep_data --root data/raw/Coronahack-Chest-XRay-Dataset

# 3. train for 10 epochs
python -m src.train --epochs 10

# 4. best model saved to checkpoints/best.pt
