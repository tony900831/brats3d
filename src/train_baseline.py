# -*- coding: utf-8 -*-
"""
Training script for BraTS 3D tumor segmentation (PyTorch + MONAI)
- Reads TRAIN_DIR from configs/.env
- Builds 80/20 split automatically and saves to configs/split_train_val.json
- 3D UNet (4 in -> 4 out), Dice + CE, AMP
- Sliding-window validation; saves best checkpoint to checkpoints/best.pt
"""

import os, json, random, math, time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    CropForegroundd, ScaleIntensityRangePercentilesd, EnsureTyped,
    ConcatItemsd, RandFlipd, RandAffined, RandSpatialCropSamplesd,
    Lambdad, AsDiscreted
)

# ------------------------- utilities -------------------------
ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "configs" / ".env"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR  = ROOT / "outputs"
CFG_SPLIT = ROOT / "configs" / "split_train_val.json"

def read_env(env_path: Path) -> Dict[str, str]:
    kv = {}
    if not env_path.exists():
        raise FileNotFoundError(f"Missing {env_path}. Please create configs/.env")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith("#"): 
            continue
        k,v = line.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv

def scan_brats_cases(train_dir: str) -> List[Dict]:
    train_dir = Path(train_dir)
    items = []
    for case_dir in sorted(train_dir.glob("BraTS20*")):
        cid = case_dir.name
        flair = case_dir / f"{cid}_flair.nii.gz"
        t1    = case_dir / f"{cid}_t1.nii.gz"
        t1ce  = case_dir / f"{cid}_t1ce.nii.gz"
        t2    = case_dir / f"{cid}_t2.nii.gz"
        seg   = case_dir / f"{cid}_seg.nii.gz"
        if all(p.exists() for p in [flair, t1, t1ce, t2, seg]):
            items.append({
                "case_id": cid,
                "flair": str(flair),
                "t1":    str(t1),
                "t1ce":  str(t1ce),
                "t2":    str(t2),
                "seg":   str(seg),
            })
    if not items:
        raise RuntimeError(f"No BraTS cases found under {train_dir}")
    return items

def make_split(items: List[Dict], seed=42, val_ratio=0.2) -> Tuple[List[Dict], List[Dict]]:
    random.Random(seed).shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    val_items = items[:n_val]
    train_items = items[n_val:]
    return train_items, val_items

def label_remap(x: torch.Tensor) -> torch.Tensor:
    # seg labels: 0,1,2,4 -> 0,1,2,3
    x = x.clone()
    x[x==4] = 3
    return x

def build_transforms(voxel=(1,1,1), patch=(128,128,128), train=True):
    keys_img = ["flair","t1","t1ce","t2"]
    keys_all = keys_img + ["seg"]
    base = [
        LoadImaged(keys=keys_all, image_only=False),
        EnsureChannelFirstd(keys=keys_all),                 # -> (1,H,W,D)
        Spacingd(keys=keys_all, pixdim=voxel, mode=("bilinear","bilinear","bilinear","bilinear","nearest")),
        Orientationd(keys=keys_all, axcodes="RAS"),
        CropForegroundd(keys=keys_all, source_key="flair"), # crop to brain
        ScaleIntensityRangePercentilesd(keys=keys_img, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys="seg", func=label_remap),
        ConcatItemsd(keys=keys_img, name="image", dim=0),   # image: (4,H,W,D)
        EnsureTyped(keys=["image","seg"]),
    ]
    if train:
        aug = [
            RandFlipd(keys=["image","seg"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image","seg"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image","seg"], prob=0.5, spatial_axis=2),
            RandAffined(keys=["image","seg"], prob=0.15, rotate_range=(0.1,0.1,0.1), scale_range=(0.1,0.1,0.1), mode=("bilinear","nearest")),
            RandSpatialCropSamplesd(keys=["image","seg"], roi_size=patch, num_samples=1, random_size=False),
        ]
        return Compose(base + aug)
    else:
        return Compose(base)

def build_loaders(train_items, val_items, voxel=(1,1,1), patch=(128,128,128), batch=1, workers=4):
    train_tf = build_transforms(voxel, patch, train=True)
    val_tf   = build_transforms(voxel, patch, train=False)

    train_ds = CacheDataset(train_items, transform=train_tf, cache_rate=0.05, num_workers=workers)
    val_ds   = Dataset(val_items, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=list_data_collate)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def build_model():
    net = UNet(
        spatial_dims=3,
        in_channels=4, out_channels=4,
        channels=(16,32,64,128), strides=(2,2,2), num_res_units=2
    )
    return net

# ------------------------- training -------------------------
def train():
    env = read_env(ENV_PATH)
    TRAIN_DIR = env["TRAIN_DIR"]
    voxel = tuple(map(float, (env.get("VOXEL_SPACING","1,1,1").split(","))))
    patch = tuple(map(int, (env.get("PATCH_SIZE","128,128,128").split(","))))

    items = scan_brats_cases(TRAIN_DIR)
    if CFG_SPLIT.exists():
        split = json.loads(CFG_SPLIT.read_text(encoding="utf-8"))
        train_items = [it for it in items if it["case_id"] in split["train_ids"]]
        val_items   = [it for it in items if it["case_id"] in split["val_ids"]]
    else:
        train_items, val_items = make_split(items, seed=42, val_ratio=0.2)
        CFG_SPLIT.write_text(json.dumps({
            "train_ids": [it["case_id"] for it in train_items],
            "val_ids":   [it["case_id"] for it in val_items],
        }, indent=2), encoding="utf-8")
        print(f"[Split] train={len(train_items)}  val={len(val_items)}  -> {CFG_SPLIT}")

    train_loader, val_loader = build_loaders(train_items, val_items, voxel=voxel, patch=patch, batch=1, workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model().to(device)

    criterion = DiceCELoss(softmax=True, to_onehot_y=True, squared_pred=True, include_background=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4)
    scaler = GradScaler(enabled=True)

    dice_metric = DiceMetric(include_background=False, reduction="mean")      # overall (exclude bg)
    dice_metric_perclass = DiceMetric(include_background=False, reduction="none")

    max_epochs = int(env.get("MAX_EPOCHS", "50"))
    val_every = 1
    best_dice = -1.0
    patience = 10
    no_improve = 0

    CKPT_DIR.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, max_epochs+1):
        net.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            x = batch["image"].to(device)     # (B,4,128,128,128)
            y = batch["seg"].to(device)       # (B,1,H,W,D)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                logits = net(x)               # (B,4,H,W,D)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach().item()

        avg_loss = epoch_loss / max(1, len(train_loader))
        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] train loss={avg_loss:.4f}  ({dt:.1f}s)")

        # -------- validation --------
        if epoch % val_every == 0:
            net.eval()
            dice_metric.reset()
            dice_metric_perclass.reset()

            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device)     # full volume
                    y = batch["seg"].to(device)
                    # sliding window
                    roi_size = patch
                    sw_batch = 1
                    pred = sliding_window_inference(x, roi_size, sw_batch, net, overlap=0.5, mode="gaussian")
                    # discretize
                    post_pred = AsDiscreted(argmax=True, to_onehot=4)(pred)
                    post_label = AsDiscreted(to_onehot=4)(y)
                    dice_metric(y_pred=post_pred, y=post_label)
                    dice_metric_perclass(y_pred=post_pred, y=post_label)

            mean_dice = dice_metric.aggregate().item()
            perclass = dice_metric_perclass.aggregate().detach().cpu().numpy()  # shape (3,)
            dice_metric.reset(); dice_metric_perclass.reset()
            print(f"         val Dice (overall, no-bg) = {mean_dice:.4f} | per-class (necrotic/edema/enh) = {np.array2string(perclass, precision=4)}")

            # save best
            if mean_dice > best_dice + 1e-6:
                best_dice = mean_dice
                no_improve = 0
                ckpt_path = CKPT_DIR / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "model_state": net.state_dict(),
                    "voxel": voxel,
                    "patch": patch,
                }, ckpt_path)
                print(f"         ✅ saved best checkpoint to {ckpt_path} (Dice={best_dice:.4f})")
            else:
                no_improve += 1
                print(f"         no improvement ({no_improve}/{patience})")

            if no_improve >= patience:
                print("         Early stopping.")
                break

    print(f"Training done. Best val Dice = {best_dice:.4f}")
    # also save final
    torch.save({"model_state": net.state_dict(), "voxel": voxel, "patch": patch}, CKPT_DIR / "last.pt")

if __name__ == "__main__":
    train()
