"""
Phase 3: Physics-Guided Monte Carlo Pseudo-Label Generation (2 Stains).
Preserves exact optical density manifold while estimating pixel-level uncertainty.
"""
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from importlib import import_module

LatentUNet = import_module("teacherDiffusion").LatentUNet
TrainLatentDataset = import_module("teacherDiffusion").TrainLatentDataset
BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics


def main():
    print("=== Phase 3: Physics-Guided Monte Carlo Pseudo-Label Generation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unet = LatentUNet().to(device)
    unet.load_state_dict(
        torch.load("/workspace/results/teacher_M1_MorphoStain.pth", map_location=device, weights_only=True)
    )
    unet.eval()
    
    physics = BeerLambertPhysics().to(device)
    loader = DataLoader(TrainLatentDataset(), batch_size=32, shuffle=False, num_workers=4)
    
    N_samples = 5
    timesteps = [800, 600, 400, 200, 50]
    
    with h5py.File("/workspace/pseudo_labels_train.h5", "w") as f:
        N = len(loader.dataset)
        # Dimensions changed from 3 to 2 channels for DAB and Hematoxylin
        d_mean = f.create_dataset("c_mean", shape=(N, 2, 512, 512), dtype=np.float16)
        d_var = f.create_dataset("u_var", shape=(N, 2, 512, 512), dtype=np.float16)
        
        idx = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Generating Monte Carlo Targets"):
                z_cond = batch["z_cond"].to(device)
                rgb_true = batch["rgb"].to(device)
                B = rgb_true.size(0)
                
                rgb_64 = F.interpolate(rgb_true, size=(64, 64), mode="area")
                od_64 = physics.rgb_to_od(rgb_64).view(B, 3, -1)
                
                # S_pinv is 2x3, od_64 is 3x(H*W) -> c0_init is 2x(H*W)
                c0_init = torch.matmul(physics.S_pinv, od_64).view(B, 2, 64, 64)
                
                mc_c_pixels = []
                for _ in range(N_samples):
                    x_curr = c0_init + torch.randn_like(c0_init) * 0.20
                    
                    for t_val in timesteps:
                        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
                        alpha_t = 1.0 - (t_val / 1000.0) * 0.25
                        sigma_t = (t_val / 1000.0) * 0.25
                        
                        noise_pred = unet(x_curr, t_tensor, z_cond)
                        x_curr = (x_curr - sigma_t * noise_pred) / (alpha_t + 1e-6)
                        x_curr = 0.85 * x_curr + 0.15 * c0_init
                        
                    c_pixel = F.interpolate(x_curr, size=(512, 512), mode="bilinear", align_corners=False)
                    mc_c_pixels.append(c_pixel)
                
                mc_stack = torch.stack(mc_c_pixels, dim=1)
                d_mean[idx:idx+B] = torch.mean(mc_stack, dim=1).cpu().numpy().astype(np.float16)
                d_var[idx:idx+B] = torch.var(mc_stack, dim=1).cpu().numpy().astype(np.float16)
                idx += B

        # Copy the slide_label array directly from the input h5 file
        with h5py.File("/workspace/latent_train.h5", "r") as f_in:
            if "slide_label" in f_in:
                print("\n[*] Transferring morphological routing labels...")
                f.create_dataset("slide_label", data=f_in["slide_label"][:])

    print("\n[✓] Physics-guided pseudo-labels successfully saved to '/workspace/pseudo_labels_train.h5'.")

if __name__ == "__main__":
    main()