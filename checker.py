# checker.py - Improved Multi-Target Source Separation Trainer

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import musdb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# --- 1. Configuration ---
# All hyperparameters and settings are here for easy tweaking.
Config = {
    "device": torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    "musdb_root": "musdb18/",
    "checkpoint_path": "best_model.pth",
    "targets": ["vocals", "drums", "bass", "other"],
    "n_fft": 2048,
    "hop_length": 512,
    "sample_rate": 44100,
    "chunk_duration": 4.0,  # Increased for more context
    "batch_size": 4,  # Reduced to accommodate larger model/data
    "num_epochs": 50,  # Increased for a more serious training run
    "learning_rate": 1e-4,  # Lowered for more stable training
    "weight_decay": 1e-5,  # Added for regularization
}


# --- 2. Model Architecture (SmallUNet) ---
# The model is now adapted for multi-target output.
class SmallUNet(nn.Module):
    def __init__(self, n_fft=2048, num_targets=4):
        super(SmallUNet, self).__init__()
        self.num_targets = num_targets

        # Encoder
        self.enc1 = self._conv_block(2, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = self._conv_block(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._conv_block(64, 32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._conv_block(32, 16)

        # Final layer outputs a mask for each target (stereo)
        self.final = nn.Conv2d(16, self.num_targets * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # The input to this layer now correctly matches the output of the previous one.
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x shape: [batch, 2, H, W]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        # --- THIS IS THE FIX ---
        # We need to ensure the upsampled tensor size matches the encoder tensor size
        # before concatenation, as pooling can create size mismatches.

        d4 = self.upconv4(bottleneck)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(
                d4, size=e4.shape[2:], mode="bilinear", align_corners=False
            )
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(
                d3, size=e3.shape[2:], mode="bilinear", align_corners=False
            )
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(
                d2, size=e2.shape[2:], mode="bilinear", align_corners=False
            )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[2:], mode="bilinear", align_corners=False
            )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        # --- END FIX ---

        # Output shape: [batch, num_targets * 2, H, W]
        masks = self.sigmoid(self.final(d1))

        # Reshape to: [batch, num_targets, 2, H, W]
        masks = masks.view(
            masks.size(0), self.num_targets, 2, masks.size(2), masks.size(3)
        )

        # Expand mixture to match mask shape for multiplication
        mixture_expanded = x.unsqueeze(1).expand_as(masks)

        # Multiply to get separated stems
        stems = mixture_expanded * masks

        return stems, masks


# --- 3. Dataset (More Efficient) ---
class MUSDBDataset(Dataset):
    def __init__(self, tracks, targets, chunk_duration, sample_rate, n_fft, hop_length):
        self.tracks = tracks
        self.targets = targets
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)

        self.stft = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=None, normalized=True
        )

        # Calculate total number of possible chunks without loading them
        self.num_chunks_per_track = [
            max(1, int(track.duration * self.sample_rate) - self.chunk_samples)
            // self.chunk_samples
            for track in self.tracks
        ]
        self.total_chunks = sum(self.num_chunks_per_track)
        self.track_map = []
        for i, num_chunks in enumerate(self.num_chunks_per_track):
            self.track_map.extend([i] * num_chunks)

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        track_idx = self.track_map[idx]
        track = self.tracks[track_idx]

        # Get a random chunk from the selected track
        start_sample = random.randint(
            0, int(track.duration * self.sample_rate) - self.chunk_samples
        )

        track.chunk_start = start_sample / self.sample_rate
        track.chunk_duration = self.chunk_duration

        mixture = torch.from_numpy(track.audio.T).float()

        # Load all target stems
        target_audios = [
            torch.from_numpy(track.targets[t].audio.T).float() for t in self.targets
        ]
        targets_tensor = torch.stack(target_audios)

        # Get spectrograms
        mixture_spec = self.stft(mixture)
        targets_spec = self.stft(targets_tensor)

        mixture_mag = torch.abs(mixture_spec)
        targets_mag = torch.abs(targets_spec)

        # Normalize per-chunk (simple but effective)
        max_val = mixture_mag.max()
        if max_val > 0:
            mixture_mag /= max_val
            targets_mag /= max_val

        return mixture_mag, targets_mag


# --- 4. Training and Validation Loops ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for mixture_mag, targets_mag in tqdm(loader, desc="Training"):
        mixture_mag = mixture_mag.to(device)
        targets_mag = targets_mag.to(device)

        optimizer.zero_grad()
        predicted_stems, _ = model(mixture_mag)

        loss = criterion(predicted_stems, targets_mag)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for mixture_mag, targets_mag in tqdm(loader, desc="Validating"):
            mixture_mag = mixture_mag.to(device)
            targets_mag = targets_mag.to(device)

            predicted_stems, _ = model(mixture_mag)
            loss = criterion(predicted_stems, targets_mag)
            total_loss += loss.item()

    return total_loss / len(loader)


# --- 5. Main Execution ---
def main():
    print(f"Using device: {Config['device']}")

    # Load MUSDB18 dataset
    print("Loading MUSDB18 dataset...")
    mus = musdb.DB(root=Config["musdb_root"])
    train_tracks = [t for t in mus if t.subset == "train"]
    test_tracks = [t for t in mus if t.subset == "test"]
    print(f"Train tracks: {len(train_tracks)}, Test tracks: {len(test_tracks)}")

    # Create DataLoaders
    train_dataset = MUSDBDataset(
        train_tracks,
        Config["targets"],
        Config["chunk_duration"],
        Config["sample_rate"],
        Config["n_fft"],
        Config["hop_length"],
    )
    test_dataset = MUSDBDataset(
        test_tracks,
        Config["targets"],
        Config["chunk_duration"],
        Config["sample_rate"],
        Config["n_fft"],
        Config["hop_length"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=Config["batch_size"], shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config["batch_size"], shuffle=False, num_workers=0
    )
    print(
        f"Created datasets with {len(train_dataset)} training chunks and {len(test_dataset)} validation chunks."
    )

    # Initialize model, loss, and optimizer
    model = SmallUNet(n_fft=Config["n_fft"], num_targets=len(Config["targets"])).to(
        Config["device"]
    )
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config["learning_rate"],
        weight_decay=Config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} trainable parameters.")

    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(Config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{Config['num_epochs']} ---")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config["device"]
        )
        val_loss = validate_one_epoch(model, test_loader, criterion, Config["device"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": Config,
                },
                Config["checkpoint_path"],
            )
            print(
                f"✨ New best model saved to {Config['checkpoint_path']} with validation loss {best_val_loss:.4f}"
            )

    print("\n--- Training Complete ---")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")

    # Plotting training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
