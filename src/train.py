#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BraTS2020 3D ResUNet training (v1.0.0 stable baseline)
- Residual UNet 3D + SE channel attention
- CE (label smoothing) + SoftDice(exclude background) composite loss
- Foreground-balanced patch sampling (ET boost OFF by default)
- Mixed precision (default bf16), skip non-finite loss, grad clip
- ReduceLROnPlateau scheduler + early stopping
- Validation: sliding-window with Gaussian weights + optional flip-TTA (default OFF)
- CSV logging, best/last checkpoints
- Quiet warnings; fixes negative strides after augmentation

Default configuration equals your proven ~0.74 setup:
  epochs=150, patience=20, base=32, drop=0.0, fg_ratio=0.65, et_boost=0.0,
  dice_w=[1,1,1], val_overlap=0.5, tta=0, patch=128^3, batch=1, lr=3e-4, amp=bf16
"""

import os, json, csv, time, argparse, random, warnings
from pathlib import Path
from typing import List, Tuple
import numpy as np
import nibabel as nib

# ---- quiet common warnings ----
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- Utils ----------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

def scan_cases(train_dir: Path) -> List[Path]:
    cases = []
    for d in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
        flair = next(d.glob("*_flair.nii*"), None)
        t1    = next(d.glob("*_t1.nii*"), None)
        t1ce  = next(d.glob("*_t1ce.nii*"), None)
        t2    = next(d.glob("*_t2.nii*"), None)
        seg   = next(d.glob("*_seg.nii*"), None)
        if flair and t1 and t1ce and t2 and seg:
            cases.append(d)
    return cases

def make_split(cases: List[Path], split_path: Path, val_ratio=0.2):
    ids = [c.name for c in cases]
    random.shuffle(ids); n_val = int(len(ids)*val_ratio)
    split = {"train": ids[n_val:], "val": ids[:n_val]}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f: json.dump(split, f, indent=2)
    return split

def load_split(split_path: Path): 
    with open(split_path, "r", encoding="utf-8") as f: return json.load(f)

def percent_clip(x: np.ndarray, lo=0.5, hi=99.5):
    lo_v, hi_v = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip(x, lo_v, hi_v)

def zscore(x: np.ndarray, eps=1e-5):
    m, s = np.nanmean(x), np.nanstd(x)
    x = (x - m) / (s + eps)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)

def map_seg_to_contiguous(seg: np.ndarray):
    cont = np.zeros_like(seg, dtype=np.uint8)
    cont[seg == 1] = 1
    cont[seg == 2] = 2
    cont[seg == 4] = 3
    return cont

# --------------- Dataset ---------------

class BraTSPatchDataset(Dataset):
    def __init__(self, root: Path, ids: List[str],
                 patch=(128,128,128), fg_ratio=0.65, et_boost=0.0, aug=True):
        self.root = root; self.ids = ids
        self.patch = np.array(patch); self.fg_ratio = fg_ratio
        self.et_boost = et_boost; self.aug = aug
        self.index = []
        for cid in self.ids:
            d = root / cid
            self.index.append({
                "flair": str(next(d.glob("*_flair.nii*"))),
                "t1":    str(next(d.glob("*_t1.nii*"))),
                "t1ce":  str(next(d.glob("*_t1ce.nii*"))),
                "t2":    str(next(d.glob("*_t2.nii*"))),
                "seg":   str(next(d.glob("*_seg.nii*")))
            })

    def __len__(self): return len(self.index) * 4  # oversample per epoch

    def _load_case(self, rec):
        flair = zscore(percent_clip(nib.load(rec["flair"]).get_fdata().astype(np.float32)))
        t1    = zscore(percent_clip(nib.load(rec["t1"]).get_fdata().astype(np.float32)))
        t1ce  = zscore(percent_clip(nib.load(rec["t1ce"]).get_fdata().astype(np.float32)))
        t2    = zscore(percent_clip(nib.load(rec["t2"]).get_fdata().astype(np.float32)))
        seg   = nib.load(rec["seg"]).get_fdata().astype(np.uint8)
        img = np.stack([flair, t1, t1ce, t2], axis=0)  # [4,D,H,W]
        seg = map_seg_to_contiguous(seg)               # {0..3}
        return img, seg

    def _rand_crop(self, img, seg, want_fg: bool):
        _, D, H, W = img.shape
        pd, ph, pw = self.patch
        def r(L, P): return 0 if L<=P else np.random.randint(0, L-P+1)

        if want_fg and (seg>0).any():
            # optional ET boost (default OFF)
            choose_et = (np.random.rand() < self.et_boost) and (seg==3).any()
            if choose_et:
                fgs = np.argwhere(seg==3)  # contiguous label 3 == ET
            else:
                fgs = np.argwhere(seg>0)
            zc,yc,xc = fgs[np.random.randint(0, len(fgs))]
            z0 = np.clip(zc - pd//2, 0, max(0, D-pd))
            y0 = np.clip(yc - ph//2, 0, max(0, H-ph))
            x0 = np.clip(xc - pw//2, 0, max(0, W-pw))
        else:
            z0,y0,x0 = r(D,pd), r(H,ph), r(W,pw)

        z1,y1,x1 = z0+pd, y0+ph, x0+pw
        return img[:, z0:z1, y0:y1, x0:x1], seg[z0:z1, y0:y1, x0:x1]

    def _aug(self, img, seg):
        # flips
        if np.random.rand()<0.5: img=img[:, ::-1]; seg=seg[::-1]
        if np.random.rand()<0.5: img=img[:, :, ::-1]; seg=seg[:, ::-1]
        if np.random.rand()<0.5: img=img[:, :, :, ::-1]; seg=seg[:, :, ::-1]
        # 90 deg rotations (axial)
        k=np.random.randint(0,4)
        if k: img=np.rot90(img,k,axes=(2,3)); seg=np.rot90(seg,k,axes=(1,2))
        # intensity jitter + light noise
        if np.random.rand()<0.3:
            img = img * (1.0 + np.random.uniform(-0.10,0.10)) + np.random.uniform(-0.05,0.05)
        if np.random.rand()<0.2:
            img = img + np.random.normal(0, 0.02, size=img.shape).astype(np.float32)
        return img, seg

    def __getitem__(self, idx):
        rec = self.index[idx % len(self.index)]
        img, seg = self._load_case(rec)
        img, seg = self._rand_crop(img, seg, want_fg=(np.random.rand()<self.fg_ratio))
        if self.aug: img, seg = self._aug(img, seg)
        # fix negative strides -> contiguous
        img = np.ascontiguousarray(img); seg = np.ascontiguousarray(seg)
        return torch.from_numpy(img).float(), torch.from_numpy(seg).long()

class BraTSValDataset(Dataset):
    def __init__(self, root: Path, ids: List[str]):
        self.root=root; self.ids=ids; self.index=[]
        for cid in self.ids:
            d=root/cid
            self.index.append({
                "id": cid,
                "flair": str(next(d.glob("*_flair.nii*"))),
                "t1":    str(next(d.glob("*_t1.nii*"))),
                "t1ce":  str(next(d.glob("*_t1ce.nii*"))),
                "t2":    str(next(d.glob("*_t2.nii*"))),
                "seg":   str(next(d.glob("*_seg.nii*")))
            })
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        rec=self.index[i]
        flair = zscore(percent_clip(nib.load(rec["flair"]).get_fdata().astype(np.float32)))
        t1    = zscore(percent_clip(nib.load(rec["t1"]).get_fdata().astype(np.float32)))
        t1ce  = zscore(percent_clip(nib.load(rec["t1ce"]).get_fdata().astype(np.float32)))
        t2    = zscore(percent_clip(nib.load(rec["t2"]).get_fdata().astype(np.float32)))
        seg   = nib.load(rec["seg"]).get_fdata().astype(np.uint8)
        img = np.stack([flair,t1,t1ce,t2], axis=0)
        seg = map_seg_to_contiguous(seg)
        return rec["id"], torch.from_numpy(np.ascontiguousarray(img)).float(), torch.from_numpy(np.ascontiguousarray(seg)).long()

# --------------- Model -----------------

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

# --------------- Loss/metrics ----------

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=4, smooth=1e-5, exclude_bg=True, class_weights=None):
        super().__init__()
        self.n_classes=n_classes; self.smooth=smooth
        self.exclude_bg=exclude_bg
        if class_weights is None:
            self.class_weights = None
        else:
            w = torch.tensor(class_weights, dtype=torch.float32)  # length 3 for [1,2,3]
            self.class_weights = w

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        oh = F.one_hot(target, num_classes=self.n_classes).permute(0,4,1,2,3).float()
        oh = torch.nan_to_num(oh, nan=0.0)
        if self.exclude_bg:
            probs = probs[:,1:]   # [B,3,D,H,W]
            oh = oh[:,1:]         # [B,3,D,H,W]
        dims=(0,2,3,4)
        inter = torch.sum(probs*oh, dims)
        denom = torch.sum(probs+oh, dims)
        dice_c = (2*inter + self.smooth)/(denom + self.smooth)  # [3]
        if self.class_weights is not None:
            w = self.class_weights.to(dice_c.device)
            dice_c = dice_c * w / (w.sum() / len(w))
        return 1 - dice_c.mean()

@torch.no_grad()
def dice_per_class(logits, target) -> np.ndarray:
    probs = F.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    out=[]
    for cls in [1,2,3]:
        pr=(pred==cls).float(); gt=(target==cls).float()
        inter=(pr*gt).sum(); denom=pr.sum()+gt.sum()
        out.append(((2*inter+1e-5)/(denom+1e-5)).item())
    return np.array(out, dtype=np.float32)

def estimate_ce_weights(loader, n_classes=4, max_batches=30):
    counts = torch.zeros(n_classes)
    for i, (_, y) in enumerate(loader):
        for c in range(n_classes): counts[c] += (y==c).sum().item()
        if i+1>=max_batches: break
    counts += 1
    inv = 1.0 / counts
    w = inv / inv.sum() * n_classes
    return w

# ---------- Sliding-window + TTA ----------

def pad_to_factor(x, factor=8, is_label=False):
    # x: tensor [N,C,D,H,W] or [N,D,H,W]
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
    """Return [1,1,D,H,W] smooth weighting window (max=1) to reduce seams."""
    pd, ph, pw = patch
    def gwin(L):
        x = np.linspace(-1, 1, L)
        # cos window (smooth edges)
        w = 0.5 * (1 + np.cos(np.pi * x))
        w = w / w.max()
        return w.astype(np.float32)
    wz, wy, wx = gwin(pd), gwin(ph), gwin(pw)
    w = wz[:,None,None] * wy[None,:,None] * wx[None,None,:]
    w = w / w.max()
    w = torch.from_numpy(w)[None,None,...]  # [1,1,D,H,W]
    return w

def _flip3d(t: torch.Tensor, dims: Tuple[int,...]):
    return torch.flip(t, dims=dims) if len(dims)>0 else t

@torch.no_grad()
def sliding_window_logits_tta(model, x, device, patch=(128,128,128), overlap=0.5,
                              amp_dtype=None, tta: int = 0):
    """
    x: [1,4,D,H,W] on device
    tta=0: no TTA
    tta=1: 4 views (identity, flip H, flip W, flip H+W)
    tta=2: 8 views (add flip D variants)
    """
    _,C,D,H,W = x.shape
    pd,ph,pw = patch
    sd = max(1, int(pd*(1-overlap)))
    sh = max(1, int(ph*(1-overlap)))
    sw = max(1, int(pw*(1-overlap)))

    x_pad, pads, orig = pad_to_factor(x, factor=8, is_label=False)
    _,_,D2,H2,W2 = x_pad.shape

    acc = torch.zeros((1,4,D2,H2,W2), dtype=torch.float32, device="cpu")
    cnt = torch.zeros((1,1,D2,H2,W2), dtype=torch.float32, device="cpu")

    win = gaussian_window_3d((pd,ph,pw)).to(device)  # [1,1,pd,ph,pw]

    # Define flip dims for TTA (relative to [N,C,D,H,W])
    views = [()]  # identity
    if tta >= 1:
        views += [(3,), (4,), (3,4,)]  # flip H, flip W, flip H+W
    if tta >= 2:
        views += [(2,), (2,3,), (2,4,), (2,3,4,)]  # add D flips

    ctx = torch.autocast(device_type="cuda" if x.device.type=="cuda" else "cpu",
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
                    # reverse flip back
                    logits = _flip3d(logits, dims=dims)
                    # weight by window & accumulate on CPU
                    logits_w = (logits * win).detach().to("cpu")
                    acc[..., z0:z1, y0:y1, x0:x1] += logits_w
                    cnt[..., z0:z1, y0:y1, x0:x1] += win.detach().to("cpu")

    acc = acc / torch.clamp_min(cnt, 1e-6)
    D0,H0,W0 = orig
    acc = acc[..., :D0, :H0, :W0]  # [1,4,D,H,W] on CPU
    return acc

# --------------- Train/Val --------------

def train_one_epoch(model, loader, opt, scaler, ce_loss, dice_loss, device,
                    amp_dtype=None, grad_clip=0.5):
    model.train()
    total=0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for x,y in pbar:
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu",
                             dtype=amp_dtype if amp_dtype is not None else torch.float32,
                             enabled=(amp_dtype is not None))
        with ctx:
            logits=model(x)
            loss = 0.5*ce_loss(logits,y) + 0.5*dice_loss(logits,y)

        if not torch.isfinite(loss):
            pbar.set_postfix_str("skip non-finite")
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        total += loss.item()*x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, amp_dtype=None, patch=(128,128,128), overlap=0.5, tta=0):
    model.eval()
    ce = nn.CrossEntropyLoss()
    sd = SoftDiceLoss()
    losses, dices = [], []
    pbar = tqdm(loader, desc="valid", leave=False)
    for _, x, y in pbar:
        x = x.to(device, non_blocking=True)  # [1,4,D,H,W]
        logits_cpu = sliding_window_logits_tta(
            model, x, device, patch=patch, overlap=overlap, amp_dtype=amp_dtype, tta=tta
        )
        logits = logits_cpu.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        loss = 0.5*ce(logits, y) + 0.5*sd(logits, y)
        d = dice_per_class(logits, y)

        losses.append(float(loss.item()))
        dices.append(d)
        pbar.set_postfix(dice=f"{np.mean(d):.4f}")
    if len(dices)==0:
        return 0.0, np.array([0.,0.,0.], dtype=np.float32)
    return float(np.mean(losses)), np.stack(dices).mean(axis=0)

def save_ckpt(path: Path, model, opt, epoch, best):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),"optimizer": opt.state_dict(),
                "epoch": epoch, "best_dice": best}, str(path))

# ---------------- Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--patch", type=int, nargs=3, default=[128,128,128])
    ap.add_argument("--val_patch", type=int, nargs=3, default=[128,128,128],
                    help="sliding window patch size for validation")
    ap.add_argument("--val_overlap", type=float, default=0.5)
    ap.add_argument("--tta", type=int, default=0, choices=[0,1,2])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_path", type=str, default="configs/brats/split.json")
    ap.add_argument("--force_resplit", action="store_true")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--reset_log", action="store_true")
    ap.add_argument("--amp", type=str, default="bf16", choices=["off","fp16","bf16"])
    ap.add_argument("--base", type=int, default=32, help="base channels of UNet")
    ap.add_argument("--drop", type=float, default=0.0, help="dropout in residual blocks")
    ap.add_argument("--fg_ratio", type=float, default=0.65, help="foreground patch ratio")
    ap.add_argument("--et_boost", type=float, default=0.0, help="prob to center on ET when sampling FG")
    ap.add_argument("--dice_w", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    help="Dice weights for [NCR, ED, ET]")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.train_dir)
    cases = scan_cases(root)
    print(f"[scan] found {len(cases)} training cases under {root}")

    split_path = Path(args.split_path)
    if args.force_resplit or (not split_path.exists()):
        split = make_split(cases, split_path, args.val_ratio)
        print(f"[split] regenerated split with {len(split['train'])} train / {len(split['val'])} val")
    else:
        split = load_split(split_path)
        print(f"[split] loaded split with {len(split['train'])} train / {len(split['val'])} val")

    train_ids, val_ids = split["train"], split["val"]

    train_ds = BraTSPatchDataset(root, train_ids, patch=tuple(args.patch),
                                 fg_ratio=args.fg_ratio, et_boost=args.et_boost, aug=True)
    val_ds   = BraTSValDataset(root, val_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=max(1,args.workers//2), pin_memory=True)

    model = ResUNet3D(in_ch=4, n_classes=4, base=args.base, p_drop=args.drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # CE weights (estimated quickly) + clamp; label smoothing
    ce_w = estimate_ce_weights(train_loader, n_classes=4, max_batches=30).to(device)
    ce_w = ce_w.clamp(0.1, 5.0)
    ce_loss = nn.CrossEntropyLoss(weight=ce_w, label_smoothing=0.05)

    dice_loss = SoftDiceLoss(class_weights=args.dice_w)

    amp_dtype = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    scaler = torch.amp.GradScaler("cuda") if (amp_dtype==torch.float16 and device.type=="cuda") else None

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "train_log.csv"
    if args.reset_log and log_csv.exists(): log_csv.unlink()
    if not log_csv.exists():
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","dice_overall","dice_NCR","dice_ED","dice_ET","time_sec"])

    start_epoch, best_dice = 1, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"]); 
        if "optimizer" in ckpt: opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch",0))+1
        best_dice = float(ckpt.get("best_dice",0.0))
        print(f"[resume] from {args.resume}: epoch={start_epoch-1}, best={best_dice:.4f}")

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(2, args.patience//3), min_lr=1e-6
    )

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    for epoch in range(start_epoch, args.epochs+1):
        t0=time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, opt, scaler, ce_loss, dice_loss, device,
                                     amp_dtype=amp_dtype, grad_clip=0.5)

        val_loss, dpc = validate(model, val_loader, device, amp_dtype=amp_dtype,
                                 patch=tuple(args.val_patch), overlap=args.val_overlap, tta=args.tta)
        dice_overall = float(np.nanmean(dpc))

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                                    f"{dice_overall:.4f}", f"{dpc[0]:.4f}", f"{dpc[1]:.4f}", f"{dpc[2]:.4f}",
                                    f"{time.time()-t0:.1f}"])
        print(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"Dice overall={dice_overall:.4f}  (NCR={dpc[0]:.4f}, ED={dpc[1]:.4f}, ET={dpc[2]:.4f})")

        # save last
        save_ckpt(ckpt_dir/"last.pt", model, opt, epoch, best_dice)

        improved = dice_overall > best_dice
        if improved:
            best_dice = dice_overall
            save_ckpt(best_path, model, opt, epoch, best_dice)
            print(f"  ↑ new best: {best_dice:.4f}  (saved: {best_path})")

        sched.step(dice_overall)

        # early stopping by patience on "no improvement"
        if not improved:
            patience_counter = getattr(main, "_p", 0) + 1
            setattr(main, "_p", patience_counter)
            print(f"  no improvement ({patience_counter})")
            if patience_counter >= args.patience:
                print("Early stopping.")
                break
        else:
            setattr(main, "_p", 0)

    print(f"\nTraining done. Best Dice={best_dice:.4f}")
    print(f"[hint] Best checkpoint at: {best_path}")

if __name__ == "__main__":
    main()
