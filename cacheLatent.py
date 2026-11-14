"""
Phase 1: Strict Split-Preserving Dual Compression.
Extracts VAE latents, Phikon semantics, and slide labels (AT vs I6) for Train, Val, and Test folders.
"""
import os, h5py, torch
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from transformers import AutoModel

class WsiPatchDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.vae_tf = transforms.Compose([
            transforms.Resize((512, 512)), transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.vit_tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.raw_tf = transforms.Compose([
            transforms.Resize((512, 512)), transforms.ToTensor()
        ])

    def __len__(self): 
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        filename = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        
        # Extract Morphological Routing Label based on filename
        # 1.0 for Tau (AT8), 0.0 for Amyloid-Beta (6E10)
        label = 1.0 if "-AT" in filename else 0.0
        
        return self.vae_tf(img), self.vit_tf(img), self.raw_tf(img), filename, label

def process_split(split_name, device, vae, vit):
    search_path = f"/workspace/dataset/{split_name}/**/*.jpg"
    paths = sorted(glob(search_path, recursive=True))
    if not paths:
        print(f"[!] Skip: No images in {split_name}")
        return

    print(f"\n[*] Caching '{split_name}' split ({len(paths)} patches)...")
    loader = DataLoader(WsiPatchDataset(paths), batch_size=16, num_workers=4)
    out_h5 = f"/workspace/latent_{split_name}.h5"
    
    with h5py.File(out_h5, "w") as h5f:
        N = len(paths)
        d_lat = h5f.create_dataset("latents", shape=(N, 4, 64, 64), dtype=np.float32)
        d_zcond = h5f.create_dataset("z_cond", shape=(N, 1, 768), dtype=np.float32)
        d_rgb = h5f.create_dataset("rgb", shape=(N, 3, 512, 512), dtype=np.float32)
        d_label = h5f.create_dataset("slide_label", shape=(N,), dtype=np.float32)

        idx = 0
        for vae_img, vit_img, raw_rgb, _, labels in tqdm(loader):
            vae_img, vit_img = vae_img.to(device), vit_img.to(device)
            bsz = vae_img.size(0)
            
            with torch.no_grad():
                latents = vae.encode(vae_img).latent_dist.sample() * 0.18215
                z_cond = vit(vit_img).last_hidden_state[:, 0:1, :]
            
            d_lat[idx:idx+bsz] = latents.cpu().numpy()
            d_zcond[idx:idx+bsz] = z_cond.cpu().numpy()
            d_rgb[idx:idx+bsz] = raw_rgb.numpy()
            d_label[idx:idx+bsz] = labels.numpy()
            idx += bsz
            
    print(f"[✓] {split_name}.h5 saved.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("/workspace/results", exist_ok=True)
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    vit = AutoModel.from_pretrained("owkin/phikon").to(device).eval()
    for p in vae.parameters(): p.requires_grad = False
    for p in vit.parameters(): p.requires_grad = False

    for split in ["train", "val", "test"]:
        process_split(split, device, vae, vit)

if __name__ == "__main__": 
    main()