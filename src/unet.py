"""U-Net (PyTorch) do segmentacji naczyń — wymagania na 5.0, zbiór HRF.

Sieć uczy się na CAŁYCH obrazach (nie wycinkach). Obrazy HRF skalujemy w dół
(scale=0.2 → 467×701) i dopełniamy do wielokrotności 16 (480×704), bo U-Net cztery
razy zmniejsza rozdzielczość. Stratę liczymy tylko wewnątrz pola widzenia (FOV).

Akceleracja: Apple MPS (Metal) jeśli dostępna, inaczej CPU.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .data import load_hrf_image, load_hrf_fov, load_hrf_label

ORIG_H, ORIG_W = 467, 701      # rozmiar obrazu HRF przy scale=0.2
PAD_H, PAD_W = 480, 704        # wielokrotności 16


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad(arr: np.ndarray) -> np.ndarray:
    """Dopełnia (H,W) lub (H,W,C) z (467,701) do (480,704) odbiciem na krawędziach."""
    ph, pw = PAD_H - ORIG_H, PAD_W - ORIG_W
    pad = [(0, ph), (0, pw)] + ([(0, 0)] if arr.ndim == 3 else [])
    return np.pad(arr, pad, mode="reflect")


# --- model --------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        b = base
        self.d1 = _DoubleConv(in_ch, b)
        self.d2 = _DoubleConv(b, b * 2)
        self.d3 = _DoubleConv(b * 2, b * 4)
        self.d4 = _DoubleConv(b * 4, b * 8)
        self.bott = _DoubleConv(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.u4 = _DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.u3 = _DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.u2 = _DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.u1 = _DoubleConv(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        bn = self.bott(self.pool(c4))
        x = self.u4(torch.cat([self.up4(bn), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.out(x)


# --- dane ---------------------------------------------------------------------
class HRFDataset(Dataset):
    """Zwraca (obraz 3×H×W, maska 1×H×W, fov 1×H×W) — dopełnione do 480×704."""

    def __init__(self, ids, scale=0.2, augment=False):
        self.items = []
        for i in ids:
            rgb = _pad(load_hrf_image(i, scale)).astype(np.float32) / 255.0
            gt = _pad(load_hrf_label(i, scale)).astype(np.float32)
            fov = _pad(load_hrf_fov(i, scale)).astype(np.float32)
            self.items.append((rgb, gt, fov))
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb, gt, fov = self.items[idx]
        if self.augment:
            if np.random.rand() < 0.5:  # poziome odbicie
                rgb, gt, fov = rgb[:, ::-1], gt[:, ::-1], fov[:, ::-1]
            if np.random.rand() < 0.5:  # pionowe odbicie
                rgb, gt, fov = rgb[::-1], gt[::-1], fov[::-1]
        rgb = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        gt = torch.from_numpy(np.ascontiguousarray(gt))[None]
        fov = torch.from_numpy(np.ascontiguousarray(fov))[None]
        return rgb, gt, fov


def _masked_loss(logits, target, fov):
    """BCE + Dice, liczone tylko wewnątrz FOV."""
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * fov).sum() / fov.sum().clamp_min(1)
    p, t = prob * fov, target * fov
    inter = (p * t).sum(dim=[1, 2, 3])
    dice = 1 - (2 * inter + 1) / (p.sum(dim=[1, 2, 3]) + t.sum(dim=[1, 2, 3]) + 1)
    return bce + dice.mean()


def train_unet(train_ids, scale=0.2, epochs=60, batch_size=4, lr=1e-3, base=16,
               seed=0, device=None, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or get_device()
    loader = DataLoader(HRFDataset(train_ids, scale, augment=True),
                        batch_size=batch_size, shuffle=True)
    model = UNet(base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for rgb, gt, fov in loader:
            rgb, gt, fov = rgb.to(device), gt.to(device), fov.to(device)
            opt.zero_grad()
            loss = _masked_loss(model(rgb), gt, fov)
            loss.backward(); opt.step()
            tot += loss.item()
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  epoka {ep:3d}/{epochs}  strata={tot/len(loader):.4f}")
    return model


@torch.no_grad()
def predict_mask(model, rgb, fov, threshold=0.5, device=None):
    """Predykcja dla jednego obrazu HRF (scale=0.2) → maska bool (467×701) w FOV."""
    device = device or get_device()
    model.eval()
    x = _pad(rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    x = torch.from_numpy(np.ascontiguousarray(x))[None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    prob = prob[:ORIG_H, :ORIG_W]
    return (prob > threshold) & (np.asarray(fov) > 0)
