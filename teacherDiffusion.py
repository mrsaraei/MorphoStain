"""
Phase 2: Physics-Guided Latent Diffusion Training (MorphoStain Teacher).
Incorporates the calibrated 3x2 H-DAB Optical Density matrix for SEA-AD brightfield WSIs.
"""
import os
import csv
import argparse
import random
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class BeerLambertPhysics(nn.Module):
    """
    Implements forward and inverse Beer-Lambert optical density transformations.
    Uses a calibrated 3x2 absorbance matrix for DAB (Brown) and Hematoxylin (Blue).
    """
    def __init__(self):
        super().__init__()
        self.I0 = 255.0
        # Calibrated 3x2 Matrix:
        # Col 0 -> DAB (Brown: absorbs Blue > Green > Red)
        # Col 1 -> Hematoxylin (Blue: absorbs Red > Green > Blue)
        S_init = torch.tensor([
            [0.268, 0.776],  # Red absorbance
            [0.570, 0.550],  # Green absorbance (Fixed row order)
            [0.776, 0.286]   # Blue absorbance  (Fixed row order)
        ], dtype=torch.float32)
        
        S_norm = F.normalize(S_init, p=2, dim=0)
        self.register_buffer("S_AD", S_norm)
        self.register_buffer("S_pinv", torch.linalg.pinv(S_norm))

    def rgb_to_od(self, rgb: torch.Tensor) -> torch.Tensor:
        """Converts normalized RGB [0, 1] to Optical Density (OD)."""
        rgb_clamped = torch.clamp(rgb, 1e-5, 1.0)
        return -torch.log10(rgb_clamped)

    def rgb_to_conc_pinv(self, rgb: torch.Tensor) -> torch.Tensor:
        """Projects RGB onto the 2-stain concentration subspace via Moore-Penrose pseudoinverse."""
        B, _, H, W = rgb.shape
        od = self.rgb_to_od(rgb)
        return torch.matmul(self.S_pinv, od.reshape(B, 3, -1)).reshape(B, 2, H, W)

    def conc_to_rgb(self, c: torch.Tensor) -> torch.Tensor:
        """Reconstructs normalized RGB [0, 1] from 2-channel stain concentrations."""
        B, C, H, W = c.shape
        od = torch.matmul(self.S_AD, c.reshape(B, C, -1)).reshape(B, 3, H, W)
        rgb = self.I0 * torch.pow(10.0, -torch.clamp(od, min=0.0)) / 255.0
        return torch.clamp(rgb, 0.0, 1.0)


class LatentUNet(nn.Module):
    """Conditional Latent Diffusion UNet operating in 64x64 stain concentration space."""
    def __init__(self, disable_phikon: bool = False):
        super().__init__()
        self.disable_phikon = disable_phikon
        self.cond_proj = nn.Linear(768, 128)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        self.enc = nn.Sequential(
            nn.Conv2d(2, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        self.mid = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        self.out = nn.Conv2d(128, 2, 3, padding=1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        feat = self.enc(x)
        
        t_norm = (t.float() / 1000.0).view(-1, 1)
        t_emb = self.time_mlp(t_norm).unsqueeze(-1).unsqueeze(-1)
        feat = feat + t_emb
        
        if not self.disable_phikon and z_cond is not None:
            if z_cond.dim() == 3:
                z_cond = z_cond.squeeze(1)
            c_emb = self.cond_proj(z_cond).unsqueeze(-1).unsqueeze(-1)
            feat = feat + c_emb
            
        feat = feat + self.mid(feat)
        return self.out(feat)


class TrainLatentDataset(Dataset):
    def __init__(self): 
        self.h5 = h5py.File("/workspace/latent_train.h5", "r")
        self.length = self.h5["latents"].shape[0]

    def __len__(self): 
        return self.length

    def __getitem__(self, i: int):
        return {
            "z_cond": torch.from_numpy(self.h5["z_cond"][i]),
            "rgb": torch.from_numpy(self.h5["rgb"][i])
        }


def apply_geometric_augmentations(rgb: torch.Tensor) -> torch.Tensor:
    """Applies random rotations and flips to the RGB tensor batch."""
    if random.random() > 0.5:
        rgb = torch.flip(rgb, dims=[-1])
    if random.random() > 0.5:
        rgb = torch.flip(rgb, dims=[-2])
    k_rots = random.randint(0, 3)
    if k_rots > 0:
        rgb = torch.rot90(rgb, k=k_rots, dims=[-2, -1])
    return rgb


def main():
    print("=== Phase 2: Training MorphoStain Physics-Guided Teacher Diffusion Model ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="M1_MorphoStain")
    parser.add_argument("--disable_phikon", action="store_true")
    parser.add_argument("--lambda_inv", type=float, default=0.5)
    parser.add_argument("--lambda_ov", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs("/workspace/results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unet = LatentUNet(disable_phikon=args.disable_phikon).to(device)
    physics = BeerLambertPhysics().to(device)
    
    loader = DataLoader(TrainLatentDataset(), batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=2e-4, weight_decay=1e-5)
    
    csv_path = f"/workspace/results/loss_{args.run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss"])

    for epoch in range(1, 11):
        unet.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/10 [{args.run_name}]")
        
        for batch in pbar:
            z_cond = batch["z_cond"].to(device)
            rgb_true = apply_geometric_augmentations(batch["rgb"].to(device))
            B = rgb_true.size(0)
            
            with torch.no_grad():
                rgb_64 = F.interpolate(rgb_true, size=(64, 64), mode="area")
                od_64 = physics.rgb_to_od(rgb_64).reshape(B, 3, -1)
                c0_lat = torch.matmul(physics.S_pinv, od_64).reshape(B, 2, 64, 64)
            
            t = torch.randint(0, 1000, (B,), device=device)
            alpha_t = (1.0 - (t.float() / 1000.0) * 0.25).view(B, 1, 1, 1)
            sigma_t = ((t.float() / 1000.0) * 0.25).view(B, 1, 1, 1)
            
            noise = torch.randn_like(c0_lat)
            x_t = alpha_t * c0_lat + sigma_t * noise
            
            optimizer.zero_grad(set_to_none=True)
            
            noise_pred = unet(x_t, t, z_cond)
            loss_dm = F.mse_loss(noise_pred, noise)
            
            c0_pred = (x_t - sigma_t * noise_pred) / (alpha_t + 1e-6)
            c_pixel = F.interpolate(c0_pred, size=(512, 512), mode="bilinear", align_corners=False)
            rgb_recon = physics.conc_to_rgb(c_pixel)
            
            loss_recon = F.l1_loss(rgb_recon, rgb_true)
            loss_inv = F.l1_loss(
                -torch.log10(1.0 - rgb_recon + 1e-5),
                -torch.log10(1.0 - rgb_true + 1e-5)
            ) * args.lambda_inv
            
            c_pos = F.relu(c_pixel)
            loss_ov = torch.mean(F.relu(c_pos[:, 0] - 0.12) * F.relu(c_pos[:, 1] - 0.12)) * args.lambda_ov
            
            loss = loss_dm + (10.0 * loss_recon) + loss_inv + loss_ov
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(loader)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_loss])
            
        torch.save(unet.state_dict(), f"/workspace/results/teacher_{args.run_name}.pth")
        
    print(f"[✓] Teacher Model '{args.run_name}' trained and saved successfully.")


if __name__ == "__main__":
    main()