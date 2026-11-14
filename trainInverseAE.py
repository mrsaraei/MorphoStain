"""
InverseAE: Unsupervised Deep Stain Decomposition via Inversion Regulation (MIDL 2023).
Benchmark Implementation for MorphoStain Comparison.
Trains on /workspace/latent_train.h5 using:
1. Forward Optical Density Reconstruction Loss (Beer-Lambert)
2. Properly Anchored Inverse OD Regulation Loss (Zero-Background Preserving)
Saves:
- /workspace/results/best_inverse_ae.pth
- /workspace/results/loss_InverseAE.csv
"""

import os
import csv
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from importlib import import_module


class InverseAEModel(nn.Module):
    """
    Conditional Convolutional Auto-Encoder for Stain Concentration Map Estimation.
    """
    def __init__(self, in_channels: int = 3, num_stains: int = 2):
        super().__init__()
        self.num_stains = num_stains

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, num_stains, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        c_pred = self.dec1(torch.cat([d2, e1], dim=1))
        return c_pred


class LatentH5Dataset(Dataset):
    def __init__(self, h5_path: str = "/workspace/latent_train.h5"):
        self.h5_file = h5py.File(h5_path, "r")
        self.rgb_data = self.h5_file["rgb"]

    def __len__(self):
        return self.rgb_data.shape[0]

    def __getitem__(self, idx: int):
        rgb = torch.from_numpy(self.rgb_data[idx]).float()
        return rgb


def compute_inverse_regulation_loss(
    rgb_true: torch.Tensor,
    c_pred: torch.Tensor,
    physics: nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Forward Beer-Lambert Reconstruction Loss and Anchored Inversion Regulation Loss.
    Anchoring ensures zero stain concentration corresponds to white background in both spaces.
    """
    B, _, H, W = rgb_true.shape

    # 1. Forward Optical Density & Reconstruction
    od_true = physics.rgb_to_od(rgb_true).reshape(B, 3, -1)
    od_pred = torch.matmul(physics.S_AD, c_pred.reshape(B, 2, -1))
    loss_recon = F.mse_loss(od_pred, od_true)

    # 2. Anchored Inversion Regulation (Complementary Absorption Manifold)
    # Using complementary transmittance relative to max optical density cap (OD_max = 2.5)
    od_cap = 2.5
    od_inv_true = torch.clamp(od_cap - od_true, min=0.0)
    
    # Complementary concentration with respect to tissue saturation bound
    c_cap = 2.0
    c_inv_pred = torch.clamp(c_cap - c_pred.reshape(B, 2, -1), min=0.0)
    
    # Project complementary concentrations using normalized stain basis
    s_norm = F.normalize(physics.S_AD, p=1, dim=1)
    od_inv_pred = torch.matmul(s_norm, c_inv_pred) * (od_cap / c_cap)
    
    loss_inverse = F.mse_loss(od_inv_pred, od_inv_true)

    # Total loss with balanced regularization weight
    total_loss = loss_recon + 0.15 * loss_inverse
    return total_loss, loss_recon, loss_inverse


def main():
    print("=" * 70)
    print(" TRAINING INVERSE-AE (SOTA DEEP LEARNING BASELINE - MIDL 2023)")
    print("=" * 70)

    os.makedirs("/workspace/results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics
    physics = BeerLambertPhysics().to(device)

    model = InverseAEModel(in_channels=3, num_stains=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    dataset = LatentH5Dataset("/workspace/latent_train.h5")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    csv_log = "/workspace/results/loss_InverseAE.csv"
    with open(csv_log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss_total", "loss_recon", "loss_inverse"])

    scaler = torch.amp.GradScaler("cuda")
    best_loss = float("inf")

    for epoch in range(1, 11):
        model.train()
        total_epoch_loss = 0.0
        recon_epoch_loss = 0.0
        inv_epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"InverseAE Epoch {epoch}/10")
        for rgb in pbar:
            rgb = rgb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                c_pred = model(rgb)
                loss, l_rec, l_inv = compute_inverse_regulation_loss(rgb, c_pred, physics)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_epoch_loss += loss.item()
            recon_epoch_loss += l_rec.item()
            inv_epoch_loss += l_inv.item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Rec": f"{l_rec.item():.4f}"})

        scheduler.step()
        n_batches = len(loader)
        avg_loss = total_epoch_loss / n_batches

        with open(csv_log, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                round(avg_loss, 5),
                round(recon_epoch_loss / n_batches, 5),
                round(inv_epoch_loss / n_batches, 5)
            ])

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "/workspace/results/best_inverse_ae.pth")

    print("\n[✓] InverseAE training finished successfully.")
    print("[✓] Checkpoint saved: '/workspace/results/best_inverse_ae.pth'")
    print(f"[✓] Loss curves exported: '{csv_log}'")


if __name__ == "__main__":
    main()