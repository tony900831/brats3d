# -*- coding: utf-8 -*-
"""
Standalone validator / exporter
- Loads checkpoints/best.pt
- Evaluates on the saved validation split (configs/split_train_val.json)
- Optionally saves predicted masks to outputs/preds/*.nii.gz
"""
import json
from pathlib import Path
import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    CropForegroundd, ScaleIntensityRangePercentilesd, EnsureTyped,
    ConcatItemsd, Lambdad, AsDiscreted
)
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "configs" / ".env"
SPLIT = ROOT / "configs" / "split_train_val.json"
CKPT = ROOT / "checkpoints" / "best.pt"
OUT_DIR = ROOT / "outputs" / "preds"

def read_env(p: Path):
    kv={}
    for line in p.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        k,v=line.split("=",1)
        kv[k.strip()] = v.strip()
    return kv

def label_remap(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[x==4] = 3
    return x

def build_transforms(voxel=(1,1,1)):
    keys_img = ["flair","t1","t1ce","t2"]
    keys_all = keys_img + ["seg"]
    return Compose([
        LoadImaged(keys=keys_all, image_only=False),
        EnsureChannelFirstd(keys=keys_all),
        Spacingd(keys=keys_all, pixdim=voxel, mode=("bilinear","bilinear","bilinear","bilinear","nearest")),
        Orientationd(keys=keys_all, axcodes="RAS"),
        CropForegroundd(keys=keys_all, source_key="flair"),
        ScaleIntensityRangePercentilesd(keys=keys_img, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys="seg", func=label_remap),
        ConcatItemsd(keys=keys_img, name="image", dim=0),
        EnsureTyped(keys=["image","seg"]),
    ])

def build_items(train_dir: str, val_ids):
    items=[]
    tdir = Path(train_dir)
    for cid in val_ids:
        d = tdir/cid
        items.append({
            "case_id": cid,
            "flair": str(d/f"{cid}_flair.nii.gz"),
            "t1": str(d/f"{cid}_t1.nii.gz"),
            "t1ce": str(d/f"{cid}_t1ce.nii.gz"),
            "t2": str(d/f"{cid}_t2.nii.gz"),
            "seg": str(d/f"{cid}_seg.nii.gz"),
        })
    return items

def main(save_preds=True):
    env = read_env(ENV_PATH)
    voxel = tuple(map(float, (env.get("VOXEL_SPACING","1,1,1").split(","))))
    TRAIN_DIR = env["TRAIN_DIR"]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    val_ids = split["val_ids"]

    ds = Dataset(build_items(TRAIN_DIR, val_ids), transform=build_transforms(voxel))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # build model & load weights
    ckpt = torch.load(CKPT, map_location="cpu")
    net = UNet(spatial_dims=3, in_channels=4, out_channels=4,
               channels=(16,32,64,128), strides=(2,2,2), num_res_units=2)
    net.load_state_dict(ckpt["model_state"])
    net = net.cuda() if torch.cuda.is_available() else net
    net.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_pc = DiceMetric(include_background=False, reduction="none")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].cuda() if torch.cuda.is_available() else batch["image"]
            y = batch["seg"].cuda() if torch.cuda.is_available() else batch["seg"]
            cid = batch.get("case_id", ["unknown"])[0] if isinstance(batch, dict) else "unknown"
            roi = tuple(int(v) for v in ckpt.get("patch", (128,128,128)))
            pred = sliding_window_inference(x, roi, 1, net, overlap=0.5, mode="gaussian")
            post_pred = AsDiscreted(argmax=True, to_onehot=4)(pred)
            post_label = AsDiscreted(to_onehot=4)(y)
            dice_metric(y_pred=post_pred, y=post_label)
            dice_metric_pc(y_pred=post_pred, y=post_label)

            if save_preds:
                # save argmax mask (0..3) in 1mm space
                mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.int16)
                meta = batch["image_meta_dict"]
                affine = meta["affine"][0].numpy() if hasattr(meta["affine"][0], "numpy") else meta["affine"][0]
                nib.save(nib.Nifti1Image(mask, affine), OUT_DIR / f"{cid}_pred.nii.gz")

    overall = dice_metric.aggregate().item()
    perclass = dice_metric_pc.aggregate().cpu().numpy()
    print(f"[Validate] overall Dice (no-bg) = {overall:.4f}")
    print(f"[Validate] per-class Dice (necrotic/edema/enh) = {np.array2string(perclass, precision=4)}")

if __name__ == "__main__":
    main(save_preds=True)
