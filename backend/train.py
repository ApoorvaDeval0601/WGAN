"""
WGAN Training Script — CIFAR-10
Optimised for NVIDIA RTX 5060 Ti (16 GB VRAM, Blackwell arch)
ATML Lab Experiment 5 — NMIMS MPSTME

Supports resuming from a saved checkpoint automatically.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from datetime import datetime

# ──────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────
BATCH_SIZE   = 256
LR           = 0.00005
NZ           = 100
NUM_EPOCHS   = 200     # increased from 50 — resume-safe, skips already-done epochs
CRITIC_ITERS = 5
WEIGHT_CLIP  = 0.01
SAVE_DIR     = "model_output"
LOG_FILE     = os.path.join(SAVE_DIR, "training_log.json")
GEN_WEIGHTS  = os.path.join(SAVE_DIR, "generator_final.pth")
CRIT_WEIGHTS = os.path.join(SAVE_DIR, "critic_final.pth")


# ──────────────────────────────────────────
# Generator
# Input:  (B, NZ, 1, 1) noise
# Output: (B, 3, 32, 32) image in [-1, 1]
# ──────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ,  512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,   3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# ──────────────────────────────────────────
# Critic  (no sigmoid — raw Wasserstein score)
# ──────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,   128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512,   1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.main(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ──────────────────────────────────────────
# Windows requires all training code inside
# if __name__ == '__main__' to safely use
# multiprocessing for DataLoader workers.
# ──────────────────────────────────────────
if __name__ == '__main__':

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── GPU setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU found. Running on CPU.")
    print(f"✓ Device: {device}\n")

    # ── Dataset ──
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset    = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ── Models ──
    netG = Generator().to(device)
    netC = Critic().to(device)
    netG.apply(weights_init)
    netC.apply(weights_init)

    # ── Optimizers (RMSprop — WGAN paper) ──
    optimizerG = optim.RMSprop(netG.parameters(), lr=LR)
    optimizerC = optim.RMSprop(netC.parameters(), lr=LR)

    # ── Resume from checkpoint if weights exist ──
    start_epoch    = 0
    training_log   = []

    if os.path.exists(GEN_WEIGHTS) and os.path.exists(CRIT_WEIGHTS):
        print("✓ Found existing weights — resuming training...")
        netG.load_state_dict(torch.load(GEN_WEIGHTS, map_location=device))
        netC.load_state_dict(torch.load(CRIT_WEIGHTS, map_location=device))
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                training_log = json.load(f)
            start_epoch = len(training_log)
            print(f"✓ Resuming from epoch {start_epoch} (already completed: {start_epoch} epochs)\n")
    else:
        print("✓ No checkpoint found — starting fresh\n")

    if start_epoch >= NUM_EPOCHS:
        print(f"✓ Already trained {start_epoch} epochs which meets NUM_EPOCHS={NUM_EPOCHS}.")
        print(f"  To train more, increase NUM_EPOCHS above {start_epoch}.")
        exit(0)

    # Fixed noise for consistent epoch image grids
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

    # AMP scaler — new syntax for PyTorch 2.x
    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Starting Training from epoch {start_epoch} to {NUM_EPOCHS - 1}...\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss_C, epoch_loss_G = [], []

        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device, non_blocking=True)
            cur_batch   = real_images.size(0)

            # ── Train Critic 5× per generator step ──
            for _ in range(CRITIC_ITERS):
                noise = torch.randn(cur_batch, NZ, 1, 1, device=device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    fake_images = netG(noise).detach()
                    lossC = -(torch.mean(netC(real_images)) - torch.mean(netC(fake_images)))

                optimizerC.zero_grad(set_to_none=True)
                scaler.scale(lossC).backward()
                scaler.step(optimizerC)
                scaler.update()

                # Lipschitz constraint — weight clipping
                for p in netC.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # ── Train Generator 1× ──
            noise = torch.randn(cur_batch, NZ, 1, 1, device=device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                fake_images = netG(noise)
                lossG = -torch.mean(netC(fake_images))

            optimizerG.zero_grad(set_to_none=True)
            scaler.scale(lossG).backward()
            scaler.step(optimizerG)
            scaler.update()

            epoch_loss_C.append(lossC.item())
            epoch_loss_G.append(lossG.item())

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {i}/{len(dataloader)} "
                      f"Loss C: {lossC.item():.4f}  Loss G: {lossG.item():.4f}")

        # ── Save epoch image grid ──
        with torch.no_grad():
            fake = netG(fixed_noise)
            vutils.save_image(
                fake,
                os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.png"),
                normalize=True, nrow=8
            )

        # ── Log losses ──
        avg_lossC = sum(epoch_loss_C) / len(epoch_loss_C)
        avg_lossG = sum(epoch_loss_G) / len(epoch_loss_G)
        training_log.append({
            "epoch": epoch,
            "avg_loss_C": round(avg_lossC, 4),
            "avg_loss_G": round(avg_lossG, 4),
            "timestamp": datetime.now().isoformat()
        })
        with open(LOG_FILE, "w") as f:
            json.dump(training_log, f, indent=2)

        # ── Save weights after every epoch (safe to interrupt) ──
        torch.save(netG.state_dict(), GEN_WEIGHTS)
        torch.save(netC.state_dict(), CRIT_WEIGHTS)

        print(f"\n── Epoch {epoch} done | Avg Loss C: {avg_lossC:.4f} | Avg Loss G: {avg_lossG:.4f}\n")

    print("Training finished!")