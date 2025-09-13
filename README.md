# 3D Brain MRI Tumor Segmentation (BraTS2020)

This project implements a **3D Residual UNet with Squeeze-and-Excitation (SE) blocks** for brain tumor segmentation on the BraTS2020 dataset.

## Features
- Training (`train.py`): 3D ResUNet + SE, CE+SoftDice loss, bf16 AMP, ReduceLROnPlateau, early stopping.
- Validation (`validate.py`): Sliding-window inference, computes Dice (NCR/ED/ET, WT/TC/ET), exports CSV/JSON, and visualization PNGs.
- Report (`apireport.py`): Automatically generates a concise Markdown experiment report (English + Chinese) using Gemini/OpenAI API, or offline fallback.

## Installation
```bash
pip install -r requirements.txt
