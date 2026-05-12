# DD2424_PROJECT_GROUP_11
Project for Group 11 in course DD2424 @ KTH

To download dependencies enter following in terminal:

```pip install -r requirements.txt```

HOW TO DOWNLOAD DATASET:
1. Download Transmission (used for downloading the dataset): https://transmissionbt.com
2. Download dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/ 
3. Add in a local folder and copy the path name
4. Add path name to .env (e.g. DATASET_DIR=/Users/YourName/Documents/Datasets/oxford-iiit-pet)

TRAINING MODES:
1. Default supervised run (full trainval split):
   - Leave `ENABLE_PSEUDO_LABELING` unset (or set to `false`)
   - Run `python src/main.py`

2. Semi-supervised pseudo-label experiment (issues #17 + #18):
   - Set `ENABLE_PSEUDO_LABELING=true`
   - Optional `.env` variables:
     - `LABELED_FRACTIONS=0.1,0.01`
     - `PSEUDO_LABEL_THRESHOLDS=0.7,0.8,0.9`
     - `BASELINE_EPOCHS=10`
     - `PSEUDO_EPOCHS=10`
     - `BASELINE_LR=2e-3`
     - `PSEUDO_LR=1e-3`
     - `WEIGHT_DECAY=1e-3`
     - `PSEUDO_MODEL_MODE=linear_probe` (or `finetune_layer4`)
   - Run `python src/main.py`

ARTIFACTS WRITTEN TO `results/`:
- Pseudo-label files per fraction/threshold
- Pseudo-label threshold stats and class distributions
- Model checkpoints in `results/models/`
- Training curves in `results/figures/`
- Per-class metrics CSV files (accuracy + F1)
- `semi_supervised_comparison.csv` (baseline vs pseudo-label comparison table)
- `semi_supervised_conclusion.txt` (short summary of improvement / no improvement)
