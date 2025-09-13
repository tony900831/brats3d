#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BraTS2020 Validation / Inference (v1.1.0; matches train.py baseline)
- Reports Dice for: NCR/ED/ET (contiguous 1,2,3) + WT/TC/ET (BraTS 1|2|4)
- Sliding-window logits with Gaussian weights, pad-to-8, optional flip-TTA
- Optional post-processing: remove small connected components
- Optional NIfTI export (predictions in raw labels {0,1,2,4})
- NEW: Optional visualization (--viz) — overlay masks/contours on MRI slices
- Quiet warnings
"""

import os, argparse, json, warnings
from pathlib import Path
from typing import Tuple, List
import numpy as np
import nibabel as nib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---- matplotlib for visualization (headless) ----
# 安裝: pip install matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Utils ----------

def percent_clip(x: np.ndarray, lo=0.5, hi=99.5):
    lo_v, hi_v = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip(x, lo_v, hi_v)

def zscore(x: np.ndarray, eps=1e-5):
    m, s = np.nanmean(x), np.nanstd(x)
    x = (x - m) / (s + eps)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)

def map_raw_to_cont(seg_raw: np.ndarray):
    cont = np.zeros_like(seg_raw, dtype=np.uint8)
    cont[seg_raw == 1] = 1
    cont[seg_raw == 2] = 2
    cont[seg_raw == 4] = 3
    return cont

def map_cont_to_raw(seg_cont: np.ndarray):
    raw = np.zeros_like(seg_cont, dtype=np.uint8)
    raw[seg_cont == 1] = 1
    raw[seg_cont == 2] = 2
    raw[seg_cont == 3] = 4
    return raw

def dice_bin(pred: np.ndarray, gt: np.ndarray, eps=1e-5) -> float:
    inter = float(np.sum((pred>0) & (gt>0)))
    denom = float(np.sum(pred>0) + np.sum(gt>0))
    return (2*inter + eps) / (denom + eps)

def remove_small_components_cont(seg_cont: np.ndarray,
                                 min_cc=(60, 60, 30)) -> np.ndarray:
    """
    seg_cont in {0,1,2,3}; min_cc thresholds per class [NCR, ED, ET] in voxels.
    Lazy-imports skimage to avoid hard dependency if not used.
    """
    try:
        from skimage.measure import label as cc_label
    except Exception as e:
        raise RuntimeError("Post-processing requires scikit-image. Install with: pip install scikit-image") from e

    out = seg_cont.copy()
    for cls, thr in zip([1,2,3], min_cc):
        mask = (out==cls)
        if mask.any():
            lab = cc_label(mask, connectivity=1)
            vals, counts = np.unique(lab, return_counts=True)
            for v,cnt in zip(vals, counts):
                if v==0: continue
                if cnt < thr:
                    out[lab==v] = 0
    return out

# ---------- Model (must match train.py) ----------

class SEBlock3D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(ch, ch//r, 1)
        self.fc2 = nn.Conv3d(ch//r, ch, 1)
    def forward(self, x):
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.proj = nn.Identity() if in_ch==out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.in1   = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.in2   = nn.InstanceNorm3d(out_ch, affine=True)
        self.drop  = nn.Dropout3d(p_drop) if p_drop>0 else nn.Identity()
        self.se    = SEBlock3D(out_ch, r=8)
    def forward(self,x):
        idt = self.proj(x)
        x = F.leaky_relu(self.in1(self.conv1(x)), 0.01, inplace=True)
        x = self.drop(x)
        x = self.in2(self.conv2(x))
        x = self.se(x)
        x = F.leaky_relu(x + idt, 0.01, inplace=True)
        return x

class ResUNet3D(nn.Module):
    def __init__(self, in_ch=4, n_classes=4, base=32, p_drop=0.0):
        super().__init__()
        self.e1 = ResBlock(in_ch, base, p_drop=0)
        self.d1 = nn.Conv3d(base, base*2, 2, stride=2)
        self.e2 = ResBlock(base*2, base*2, p_drop=p_drop)
        self.d2 = nn.Conv3d(base*2, base*4, 2, stride=2)
        self.e3 = ResBlock(base*4, base*4, p_drop=p_drop)
        self.d3 = nn.Conv3d(base*4, base*8, 2, stride=2)
        self.e4 = ResBlock(base*8, base*8, p_drop=p_drop)

        self.u3 = nn.ConvTranspose3d(base*8, base*4, 2, stride=2)
        self.c3 = ResBlock(base*8, base*4, p_drop=p_drop)
        self.u2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.c2 = ResBlock(base*4, base*2, p_drop=p_drop)
        self.u1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.c1 = ResBlock(base*2, base, p_drop=0)

        self.out = nn.Conv3d(base, n_classes, 1)

    def forward(self,x):
        e1=self.e1(x)
        e2=self.e2(self.d1(e1))
        e3=self.e3(self.d2(e2))
        e4=self.e4(self.d3(e3))
        d3=self.c3(torch.cat([self.u3(e4), e3], dim=1))
        d2=self.c2(torch.cat([self.u2(d3), e2], dim=1))
        d1=self.c1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)

# ---------- Inference helpers ----------

def pad_to_factor(x, factor=8, is_label=False):
    # x: [N,C,D,H,W] or [N,D,H,W]
    if x.dim()==5: N,C,D,H,W = x.shape; is_img=True
    else: N,D,H,W = x.shape; is_img=False
    need = lambda L: (factor - (L % factor)) % factor
    pd, ph, pw = need(D), need(H), need(W)
    if pd or ph or pw:
        if is_img:
            x = F.pad(x, (0,pw, 0,ph, 0,pd))
        else:
            x = F.pad(x, (0,pw, 0,ph, 0,pd), value=0 if is_label else 0.0)
    return x, (pd,ph,pw), (D,H,W)

def gaussian_window_3d(patch: Tuple[int,int,int]) -> torch.Tensor:
    pd, ph, pw = patch
    def gwin(L):
        x = np.linspace(-1, 1, L)
        w = 0.5 * (1 + np.cos(np.pi * x))
        return (w / w.max()).astype(np.float32)
    wz, wy, wx = gwin(pd), gwin(ph), gwin(pw)
    w = wz[:,None,None] * wy[None,:,None] * wx[None,None,:]
    return torch.from_numpy(w)[None,None,...]  # [1,1,D,H,W]

def _flip3d(t: torch.Tensor, dims):
    return torch.flip(t, dims=dims) if len(dims)>0 else t

@torch.no_grad()
def sliding_window_logits_tta(model, x, patch=(128,128,128), overlap=0.5, amp_dtype=None, tta=0):
    # x: [1,4,D,H,W] on device
    device = x.device
    _,_,D,H,W = x.shape
    pd,ph,pw = patch
    sd = max(1, int(pd*(1-overlap)))
    sh = max(1, int(ph*(1-overlap)))
    sw = max(1, int(pw*(1-overlap)))

    x_pad, _, orig = pad_to_factor(x, factor=8, is_label=False)
    _,_,D2,H2,W2 = x_pad.shape

    acc = torch.zeros((1,4,D2,H2,W2), dtype=torch.float32, device="cpu")
    cnt = torch.zeros((1,1,D2,H2,W2), dtype=torch.float32, device="cpu")

    win = gaussian_window_3d((pd,ph,pw)).to(device)

    views = [()]
    if tta >= 1: views += [(3,), (4,), (3,4,)]
    if tta >= 2: views += [(2,), (2,3,), (2,4,), (2,3,4,)]

    ctx = torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu",
                         dtype=amp_dtype if amp_dtype is not None else torch.float32,
                         enabled=(amp_dtype is not None))
    for dims in views:
        xv = _flip3d(x_pad, dims=dims)
        for z0 in range(0, D2 - pd + 1, sd):
            for y0 in range(0, H2 - ph + 1, sh):
                for x0 in range(0, W2 - pw + 1, sw):
                    z1,y1,x1 = z0+pd, y0+ph, x0+pw
                    with ctx:
                        logits = model(xv[..., z0:z1, y0:y1, x0:x1])  # [1,4,pd,ph,pw]
                    logits = _flip3d(logits, dims=dims)
                    logits_w = (logits * win).detach().to("cpu")
                    acc[..., z0:z1, y0:y1, x0:x1] += logits_w
                    cnt[..., z0:z1, y0:y1, x0:x1] += win.detach().to("cpu")

    acc = acc / torch.clamp_min(cnt, 1e-6)
    D0,H0,W0 = orig
    acc = acc[..., :D0, :H0, :W0]
    return acc  # CPU tensor [1,4,D,H,W]

# ---------- IO helpers ----------

def find_modalities(case_dir: Path):
    flair = next(case_dir.glob("*_flair.nii*"))
    t1    = next(case_dir.glob("*_t1.nii*"))
    t1ce  = next(case_dir.glob("*_t1ce.nii*"))
    t2    = next(case_dir.glob("*_t2.nii*"))
    seg   = next(case_dir.glob("*_seg.nii*"), None)
    return flair, t1, t1ce, t2, seg

def load_case_arrays(case_dir: Path):
    flair, t1, t1ce, t2, seg = find_modalities(case_dir)
    a_flair = nib.load(str(flair)); a_t1 = nib.load(str(t1))
    a_t1ce  = nib.load(str(t1ce));  a_t2 = nib.load(str(t2))
    img = np.stack([
        zscore(percent_clip(a_flair.get_fdata().astype(np.float32))),
        zscore(percent_clip(a_t1.get_fdata().astype(np.float32))),
        zscore(percent_clip(a_t1ce.get_fdata().astype(np.float32))),
        zscore(percent_clip(a_t2.get_fdata().astype(np.float32))),
    ], axis=0)  # [4,D,H,W]
    seg_raw = None
    if seg is not None:
        seg_raw = nib.load(str(seg)).get_fdata().astype(np.uint8)
    affine = a_flair.affine
    return img, seg_raw, affine

# ---------- Visualization helpers ----------

def _choose_bg(img: np.ndarray, modality: str) -> np.ndarray:
    """img: [4,D,H,W]; modality in {'t1ce','flair','t1','t2'}"""
    key_to_idx = {"flair":0, "t1":1, "t1ce":2, "t2":3}
    idx = key_to_idx.get(modality.lower(), 2)
    bg = img[idx]  # [D,H,W]
    # map to [0,1] for display
    vmin, vmax = np.percentile(bg, [1, 99])
    bg = np.clip((bg - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return bg

def _pick_slices(mask_3d: np.ndarray, num_slices: int) -> List[int]:
    D = mask_3d.shape[0]
    zs_pos = np.where(mask_3d > 0)[0]

    # 無腫瘤：fallback
    if zs_pos.size == 0:
        if num_slices == 1:
            return [D // 2]
        zs = np.linspace(0, max(0, D - 1), num=min(num_slices, D), dtype=int)
        return sorted(set(zs.tolist()))

    zmin, zmax = int(zs_pos.min()), int(zs_pos.max())

    # 一張：選腫瘤面積最大的那一層
    if num_slices == 1:
        areas = [int(mask_3d[z].sum()) for z in range(zmin, zmax + 1)]
        z_best = zmin + int(np.argmax(areas))
        return [z_best]

    # 多張：等間距抽樣於腫瘤範圍
    zs = np.linspace(zmin, zmax, num=min(num_slices, max(1, zmax - zmin + 1)), dtype=int)
    return sorted(set(zs.tolist()))


def save_case_viz(cid: str, img: np.ndarray, pred_cont: np.ndarray,
                  out_dir: Path, slices: int = 8, modality: str = "t1ce",
                  draw_contour: bool = True, gt_cont: np.ndarray = None,
                  viz_style: str = "both", alpha: float = 0.30):
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = _choose_bg(img, modality)  # [D,H,W]
    mask_union = (pred_cont>0) | ((gt_cont>0) if gt_cont is not None else False)
    zs = _pick_slices(mask_union, slices)

    # class colors
    line_color = {1:(1,1,0), 2:(0,1,1), 3:(1,0,0)}  # 黃/青/紅

    for z in zs:
        fig = plt.figure(figsize=(6, 6), dpi=120)
        ax = plt.axes([0,0,1,1]); ax.axis("off")
        ax.imshow(bg[z], cmap="gray", interpolation="nearest")

        for cls in [1,2,3]:
            mask = (pred_cont[z]==cls)
            if not mask.any(): continue

            if viz_style in ("overlay","both"):
                ax.imshow(np.ma.masked_where(~mask, mask), cmap=None, alpha=alpha)  # 只用透明度，不指定顏色，保留線色識別

            if viz_style in ("contour","both"):
                if draw_contour:
                    ax.contour(mask.astype(np.uint8), levels=[0.5],
                               colors=[line_color[cls]], linewidths=1.5)

        # GT 虛線輪廓（若有）
        if gt_cont is not None and viz_style in ("contour","both"):
            for cls in [1,2,3]:
                gmask = (gt_cont[z]==cls)
                if gmask.any():
                    ax.contour(gmask.astype(np.uint8), levels=[0.5],
                               colors=[line_color[cls]], linewidths=1.0, linestyles="dashed")

        ax.text(5, 18, "NCR", color=(1,1,0), fontsize=9, weight="bold")
        ax.text(45,18, "ED",  color=(0,1,1), fontsize=9, weight="bold")
        ax.text(75,18, "ET",  color=(1,0,0), fontsize=9, weight="bold")
        if gt_cont is not None and viz_style in ("contour","both"):
            ax.text(5, 36, "GT: dashed", color=(1,1,1), fontsize=8, alpha=0.8)

        plt.savefig(str(out_dir / f"{cid}_z{z:03d}.png"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="val", choices=["val","infer"],
                    help="val: use split['val'] and compute Dice if seg exists; infer: run on all cases under data_dir without requiring seg")
    ap.add_argument("--data_dir", type=str, required=True, help="Root containing cases (each with *_flair,t1,t1ce,t2 and maybe *_seg)")
    ap.add_argument("--split_path", type=str, default="configs/brats/split.json",
                    help="Required if mode=val; will use split['val']")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (best.pt)")
    ap.add_argument("--base", type=int, default=32, help="Must match training")
    ap.add_argument("--drop", type=float, default=0.0, help="Must match training")
    ap.add_argument("--amp", type=str, default="bf16", choices=["off","fp16","bf16"])
    ap.add_argument("--patch", type=int, nargs=3, default=[128,128,128])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tta", type=int, default=0, choices=[0,1,2])
    ap.add_argument("--postproc", action="store_true", help="Remove small components per class (NCR/ED=60 vox, ET=30 vox)")
    ap.add_argument("--save_pred", type=str, default="", help="Folder to save NIfTI predictions (raw labels {0,1,2,4})")
    ap.add_argument("--metrics_csv", type=str, default="", help="Optional: path to save per-case metrics CSV")
    ap.add_argument("--metrics_json", type=str, default="", help="Optional: path to save summary metrics JSON")
    # ---- NEW: visualization flags ----
    ap.add_argument("--viz", action="store_true", help="Save overlay PNGs per case")
    ap.add_argument("--viz_dir", type=str, default="outputs/viz", help="Where to save PNG overlays")
    ap.add_argument("--viz_slices", type=int, default=8, help="Number of axial slices per case")
    ap.add_argument("--viz_modality", type=str, default="t1ce", choices=["t1ce","flair","t1","t2"], help="Background modality for overlays")
    ap.add_argument("--viz_no_contour", action="store_true", help="Disable contour lines (keep filled masks)")
    ap.add_argument("--viz_style", type=str, default="both", choices=["both","contour","overlay"],
                help="可視化樣式：both=填色+輪廓, contour=只畫輪廓, overlay=只填色")
    ap.add_argument("--viz_alpha", type=float, default=0.30, help="填色透明度(overlay/both)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]

    # build & load model
    model = ResUNet3D(in_ch=4, n_classes=4, base=args.base, p_drop=args.drop).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = Path(args.data_dir)
    if args.mode == "val":
        with open(args.split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        case_ids = split["val"]
    else:
        # infer mode: all dirs under data_dir
        case_ids = [p.name for p in sorted(root.iterdir()) if p.is_dir()]

    save_dir = Path(args.save_pred) if args.save_pred else None
    if save_dir: save_dir.mkdir(parents=True, exist_ok=True)

    # storage for metrics
    per_case_rows = []
    dices_cont_all = []   # NCR/ED/ET
    dices_braTS_all = []  # WT/TC/ET

    viz_dir = Path(args.viz_dir) if args.viz else None
    if viz_dir: viz_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(case_ids, desc="valid" if args.mode=="val" else "infer")
    for cid in pbar:
        case_dir = root / cid
        try:
            img, seg_raw, affine = load_case_arrays(case_dir)
        except StopIteration:
            # missing modality; skip
            continue

        x = torch.from_numpy(np.ascontiguousarray(img))[None,...].to(device)  # [1,4,D,H,W]

        logits_cpu = sliding_window_logits_tta(
            model, x, patch=tuple(args.patch), overlap=args.overlap, amp_dtype=amp_dtype, tta=args.tta
        )  # [1,4,D,H,W] CPU
        pred_cont = torch.argmax(logits_cpu, dim=1).squeeze(0).numpy().astype(np.uint8)

        # post-processing (optional)
        if args.postproc:
            pred_cont = remove_small_components_cont(pred_cont, min_cc=(60,60,30))

        # save nifti (raw labels)
        if save_dir:
            pred_raw = map_cont_to_raw(pred_cont)
            out = nib.Nifti1Image(pred_raw.astype(np.uint8), affine)
            nib.save(out, str(save_dir / f"{cid}_pred.nii.gz"))

        # visualization (optional)
        if viz_dir:
            gt_cont = map_raw_to_cont(seg_raw) if (args.mode=="val" and seg_raw is not None) else None
            save_case_viz(
                cid=cid, img=img, pred_cont=pred_cont, out_dir=viz_dir,
                slices=args.viz_slices, modality=args.viz_modality,
                draw_contour=(not args.viz_no_contour), gt_cont=gt_cont
            )

        # metrics (only if we have GT)
        case_row = {"case": cid}
        if args.mode == "val" and seg_raw is not None:
            gt_cont = map_raw_to_cont(seg_raw)

            # (1) NCR/ED/ET
            d_ncr = dice_bin(pred_cont==1, gt_cont==1)
            d_ed  = dice_bin(pred_cont==2, gt_cont==2)
            d_et  = dice_bin(pred_cont==3, gt_cont==3)
            dices_cont_all.append([d_ncr, d_ed, d_et])
            case_row.update({"NCR": d_ncr, "ED": d_ed, "ET": d_et, "overall": (d_ncr+d_ed+d_et)/3})

            # (2) WT/TC/ET (BraTS)
            pred_raw = map_cont_to_raw(pred_cont)
            wt = dice_bin(np.isin(pred_raw, [1,2,4]), np.isin(seg_raw, [1,2,4]))
            tc = dice_bin(np.isin(pred_raw, [1,4]),    np.isin(seg_raw, [1,4]))
            et = dice_bin(pred_raw==4,                 seg_raw==4)
            dices_braTS_all.append([wt, tc, et])
            case_row.update({"WT": wt, "TC": tc, "ET_braTS": et})

        per_case_rows.append(case_row)

    # summarize
    if args.mode == "val":
        if len(dices_cont_all):
            dc = np.array(dices_cont_all)
            mean_c = dc.mean(axis=0)
            print("\n=== NCR/ED/ET (contiguous) ===")
            print(f"NCR={mean_c[0]:.4f}  ED={mean_c[1]:.4f}  ET={mean_c[2]:.4f}  | overall={mean_c.mean():.4f}")
        if len(dices_braTS_all):
            db = np.array(dices_braTS_all)
            mean_b = db.mean(axis=0)
            print("=== WT/TC/ET (BraTS) ===")
            print(f"WT ={mean_b[0]:.4f}  TC={mean_b[1]:.4f}  ET={mean_b[2]:.4f}  | mean   ={mean_b.mean():.4f}")
    else:
        print("\nInference done (no GT).")

    # save metrics if requested
    if args.metrics_csv:
        import csv
        with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({k for row in per_case_rows for k in row.keys()})
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); [w.writerow(r) for r in per_case_rows]
        print(f"[saved] CSV metrics -> {args.metrics_csv}")
    if args.metrics_json:
        summary = {}
        if len(dices_cont_all):
            dc = np.array(dices_cont_all)
            summary["NCR/ED/ET_mean"] = dc.mean(axis=0).tolist()
            summary["overall"] = float(dc.mean())
        if len(dices_braTS_all):
            db = np.array(dices_braTS_all)
            summary["WT/TC/ET_mean"] = db.mean(axis=0).tolist()
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump({"per_case": per_case_rows, "summary": summary}, f, indent=2)
        print(f"[saved] JSON metrics -> {args.metrics_json}")

if __name__ == "__main__":
    main()
