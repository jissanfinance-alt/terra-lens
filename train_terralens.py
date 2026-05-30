"""
train_terralens.py
TERRA·LENS training script — drop this in the root of terra-lens repo.

Replaces the baseline single-GFM flow with adaptive 4-GFM fusion.
Keeps the same ImprovedCompositeLoss and AdamW/scheduler setup.

Usage:
    python train_terralens.py \
        --alpha-earth-dir  /path/to/alphaearth_embeddings \
        --tessera-dir       /path/to/tessera_embeddings \
        --terramind-dir     /path/to/terramind_embeddings \
        --thor-dir          /path/to/thor_embeddings \
        --targets-dir       /path/to/labels \
        --experiment-name   terralens_run01 \
        --epochs 50 \
        --batch-size 4
"""

import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import rasterio

from core.losses import ImprovedCompositeLoss
from core.fusion import TerraLens, align_embeddings
from core.dataset import HEIGHT_NORM_CONSTANT   # reuse the same constant

# ---------------------------------------------------------------------------
# Config defaults (all overridable via CLI)
# ---------------------------------------------------------------------------

EXPERIMENT_NAME   = "terralens_run01"
BASE_DIR          = "./runs"
BATCH_SIZE        = 4       # lower than baseline — 4 embeddings per sample
PATCH_SIZE        = 256
EPOCHS            = 50
LEARNING_RATE     = 2e-4
WEIGHT_DECAY      = 1e-4
VAL_SPLIT         = 0.2
LAMBDAS           = [1.0, 0.5, 0.5, 2.0]  # MAE, SSIM, Gradient, Tversky
RANDOM_SEED       = 42

# GFM channel sizes — must match what EOTDL provides
GFM_CHANNELS = {
    "alpha_earth": 64,
    "tessera":     128,
    "terramind":   256,
    "thor":        256,
}
HIDDEN_DIM = 256

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Multi-GFM Dataset
# ---------------------------------------------------------------------------

def find_patch_ids(target_dir: str) -> list[str]:
    """Return sorted list of patch IDs from label filenames."""
    ids = []
    for f in sorted(os.listdir(target_dir)):
        if f.endswith(".tif"):
            ids.append(f.replace(".tif", ""))
    return ids


def load_tif(path: str) -> torch.Tensor:
    """Load a GeoTIFF and return a float32 tensor (C, H, W)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return torch.from_numpy(arr)


class MultiGFMDataset(Dataset):
    """
    Loads all 4 GFM embeddings + label for each patch.
    Expects one .tif per patch in each directory, named by patch ID.

    Directory layout expected:
        alpha_earth_dir/  <patch_id>.tif   (64 channels)
        tessera_dir/      <patch_id>.tif   (128 channels)
        terramind_dir/    <patch_id>.tif   (256 channels)
        thor_dir/         <patch_id>.tif   (256 channels)
        targets_dir/      <patch_id>.tif   (4 channels: seg×3 + height)
    """

    def __init__(
        self,
        patch_ids: list[str],
        alpha_earth_dir: str,
        tessera_dir: str,
        terramind_dir: str,
        thor_dir: str,
        targets_dir: str,
        is_train: bool = True,
    ):
        self.patch_ids       = patch_ids
        self.dirs            = [alpha_earth_dir, tessera_dir, terramind_dir, thor_dir]
        self.targets_dir     = targets_dir
        self.is_train        = is_train

    def __len__(self):
        return len(self.patch_ids)

    def _augment(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply identical random flip to all tensors."""
        if random.random() > 0.5:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]   # H-flip
        if random.random() > 0.5:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]   # V-flip
        return tensors

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]

        # Load all 4 GFM embeddings
        gfm_tensors = []
        for d in self.dirs:
            path = os.path.join(d, pid + ".tif")
            gfm_tensors.append(load_tif(path))

        # Load label (4-band: bld, veg, water, height)
        label = load_tif(os.path.join(self.targets_dir, pid + ".tif"))

        # Normalise height channel (band index 3)
        label[3] = label[3] / HEIGHT_NORM_CONSTANT

        # Spatial alignment — all GFMs to same H×W
        target_h, target_w = label.shape[1], label.shape[2]
        aligned = []
        for t in gfm_tensors:
            if t.shape[1] != target_h or t.shape[2] != target_w:
                t = F.interpolate(
                    t.unsqueeze(0), size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            aligned.append(t)

        # Augmentation (train only)
        if self.is_train:
            all_tensors = self._augment(aligned + [label])
            aligned = all_tensors[:-1]
            label = all_tensors[-1]

        return aligned, label   # list[4 tensors], tensor


def collate_fn(batch):
    """Custom collate: stacks list-of-4-tensors into batch."""
    gfm_list, labels = zip(*batch)
    # gfm_list: tuple of (list of 4 tensors)
    n_gfm = len(gfm_list[0])
    batched_gfms = [
        torch.stack([gfm_list[b][i] for b in range(len(gfm_list))], dim=0)
        for i in range(n_gfm)
    ]
    batched_labels = torch.stack(labels, dim=0)
    return batched_gfms, batched_labels


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_results(model, dataset, exp_dir, num_samples=3):
    model.eval()
    viz_dir = os.path.join(exp_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    names = ["% Building", "% Vegetation", "% Water", "nDSM Height (m)"]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            gfm_list, label = dataset[idx]
            inputs = [t.unsqueeze(0).to(DEVICE) for t in gfm_list]
            inputs = align_embeddings(inputs)
            out = model(inputs).squeeze().cpu().numpy()
            tgt = label.numpy()

            out[3] = out[3] * HEIGHT_NORM_CONSTANT
            tgt[3] = tgt[3] * HEIGHT_NORM_CONSTANT

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            for c in range(4):
                vmin, vmax = (0, 1) if c < 3 else (0, HEIGHT_NORM_CONSTANT)
                axes[0, c].imshow(tgt[c], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0, c].set_title(f"True {names[c]}")
                axes[0, c].axis('off')
                axes[1, c].imshow(out[c], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1, c].set_title(f"Pred {names[c]}")
                axes[1, c].axis('off')

            plt.suptitle(f"TERRA·LENS (sample {i})")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"viz_{i}.png"))
            plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train TERRA·LENS (4-GFM adaptive fusion)")
    p.add_argument("--alpha-earth-dir",  required=True)
    p.add_argument("--tessera-dir",      required=True)
    p.add_argument("--terramind-dir",    required=True)
    p.add_argument("--thor-dir",         required=True)
    p.add_argument("--targets-dir",      required=True)
    p.add_argument("--experiment-name",  default=EXPERIMENT_NAME)
    p.add_argument("--output-dir",       default=BASE_DIR)
    p.add_argument("--epochs",           type=int, default=EPOCHS)
    p.add_argument("--batch-size",       type=int, default=BATCH_SIZE)
    p.add_argument("--hidden-dim",       type=int, default=HIDDEN_DIM)
    p.add_argument("--lr",               type=float, default=LEARNING_RATE)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    exp_dir  = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "visualizations"), exist_ok=True)

    best_path = os.path.join(exp_dir, "model_best.pth")
    last_path = os.path.join(exp_dir, "model_last.pth")
    curve_path = os.path.join(exp_dir, "loss_curve.png")

    # Save config
    with open(os.path.join(exp_dir, "training_params.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"device: {DEVICE}\n")
        f.write(f"gfm_channels: {GFM_CHANNELS}\n")

    print("--- 1. Data setup ---")
    patch_ids = find_patch_ids(args.targets_dir)
    if not patch_ids:
        raise ValueError(f"No .tif files found in {args.targets_dir}")

    train_ids, val_ids = train_test_split(
        patch_ids, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    ds_kwargs = dict(
        alpha_earth_dir=args.alpha_earth_dir,
        tessera_dir=args.tessera_dir,
        terramind_dir=args.terramind_dir,
        thor_dir=args.thor_dir,
        targets_dir=args.targets_dir,
    )
    train_ds = MultiGFMDataset(train_ids, is_train=True,  **ds_kwargs)
    val_ds   = MultiGFMDataset(val_ids,   is_train=False, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")

    print("--- 2. Model init ---")
    gfm_ch = list(GFM_CHANNELS.values())
    model = TerraLens(gfm_channels=gfm_ch, hidden_dim=args.hidden_dim).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TERRA·LENS params: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = ImprovedCompositeLoss(lambdas=LAMBDAS).to(DEVICE)

    print(f"Training on {DEVICE} for {args.epochs} epochs...")
    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        run_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False)
        for gfm_batch, targets in pbar:
            inputs  = [g.to(DEVICE) for g in gfm_batch]
            targets = targets.to(DEVICE)
            inputs  = align_embeddings(inputs)

            optimizer.zero_grad()
            out = model(inputs)
            loss, *_ = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            run_loss += loss.item() * inputs[0].size(0)
            n += inputs[0].size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train = run_loss / n
        train_losses.append(epoch_train)

        # --- Val ---
        model.eval()
        val_run, val_n = 0.0, 0
        val_comp = torch.zeros(4).to(DEVICE)
        with torch.no_grad():
            for gfm_batch, targets in val_loader:
                inputs  = [g.to(DEVICE) for g in gfm_batch]
                targets = targets.to(DEVICE)
                inputs  = align_embeddings(inputs)
                out = model(inputs)
                loss, l_mae, l_ssim, l_grad, l_tv = criterion(out, targets)
                bs = inputs[0].size(0)
                val_run += loss.item() * bs
                val_comp += torch.tensor([l_mae, l_ssim, l_grad, l_tv]).to(DEVICE) * bs
                val_n += bs

        epoch_val = val_run / val_n
        val_losses.append(epoch_val)
        scheduler.step(epoch_val)

        vc = val_comp / val_n
        print(f"Epoch {epoch+1:03d} | Train {epoch_train:.4f} | Val {epoch_val:.4f} "
              f"| MAE {vc[0]:.3f} SSIM {vc[1]:.3f} Grad {vc[2]:.3f} Tv {vc[3]:.3f}")

        if epoch_val < best_val:
            best_val = epoch_val
            torch.save(model.state_dict(), best_path)
            print(f"  >> Best saved ({best_val:.4f})")

    torch.save(model.state_dict(), last_path)

    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.title("TERRA·LENS Loss Curve")
    plt.legend()
    plt.savefig(curve_path)
    plt.close()

    print("--- 3. Visualisation ---")
    visualize_results(model, val_ds, exp_dir)
    print(f"Done. Best val loss: {best_val:.4f}")
    print(f"Outputs saved to: {exp_dir}")


if __name__ == "__main__":
    main()
