import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import musdb
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class SmallUNet(nn.Module):
    def __init__(self, n_fft=2048):
        super(SmallUNet, self).__init__()

        # Encoder: half the channels of your original
        self.enc1 = self._conv_block(2, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)

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

        self.final = nn.Conv2d(16, 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e1_pool = self.pool(e1)

        e2 = self.enc2(e1_pool)
        e2_pool = self.pool(e2)

        e3 = self.enc3(e2_pool)
        e3_pool = self.pool(e3)

        e4 = self.enc4(e3_pool)
        e4_pool = self.pool(e4)

        bottleneck = self.bottleneck(e4_pool)

        d4 = self.upconv4(bottleneck)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        mask = self.sigmoid(self.final(d1))
        output = x * mask
        return output, mask

class MUSDBDataset(Dataset):
    def __init__(self, tracks, chunk_duration=2.0, n_fft=2048, hop_length=512, target='vocals', max_chunks=1000):
        self.tracks = tracks
        self.chunk_duration = chunk_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target = target
        self.sample_rate = 44100
        self.max_chunks = max_chunks  # Limit chunks for speed

        self.stft = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,
            normalized=True
        )

        self.chunk_samples = int(chunk_duration * self.sample_rate)

        # Precalculate all chunks but limit to max_chunks
        all_chunks = []
        for track in tracks:
            track_samples = int(track.duration * self.sample_rate)
            n_chunks = max(1, (track_samples - self.chunk_samples) // (self.chunk_samples // 2))
            for i in range(n_chunks):
                start_sample = i * (self.chunk_samples // 2)
                all_chunks.append((track, start_sample))
        if len(all_chunks) > max_chunks:
            self.chunks = random.sample(all_chunks, max_chunks)
        else:
            self.chunks = all_chunks

        print(f"Created dataset with {len(self.chunks)} chunks (limited for speed)")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        track, start_sample = self.chunks[idx]

        track.chunk_start = start_sample / self.sample_rate
        track.chunk_duration = self.chunk_duration

        mixture = track.audio.T
        target_audio = track.targets[self.target].audio.T

        mixture_tensor = torch.from_numpy(mixture).float()
        target_tensor = torch.from_numpy(target_audio).float()

        mixture_spec = self.stft(mixture_tensor)
        target_spec = self.stft(target_tensor)

        mixture_mag = torch.abs(mixture_spec)
        target_mag = torch.abs(target_spec)

        mixture_mag = mixture_mag / (mixture_mag.max() + 1e-8)
        target_mag = target_mag / (target_mag.max() + 1e-8)

        return mixture_mag, target_mag, mixture_spec, target_spec

def create_data_loaders(train_tracks, test_tracks, batch_size=8, chunk_duration=2.0):
    train_dataset = MUSDBDataset(train_tracks, chunk_duration=chunk_duration, target='vocals', max_chunks=1000)
    test_dataset = MUSDBDataset(test_tracks, chunk_duration=chunk_duration, target='vocals', max_chunks=300)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=1e-3):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    model = model.to(device)

    train_losses = []
    val_losses = []

    print(f"Starting training on {device}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (mixture_mag, target_mag, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            mixture_mag = mixture_mag.to(device)
            target_mag = target_mag.to(device)

            optimizer.zero_grad()
            output, mask = model(mixture_mag)
            loss = criterion(output, target_mag)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for mixture_mag, target_mag, _, _ in test_loader:
                mixture_mag = mixture_mag.to(device)
                target_mag = target_mag.to(device)
                output, mask = model(mixture_mag)
                loss = criterion(output, target_mag)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

        # Save checkpoint each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, f'checkpoint_epoch_{epoch+1}.pth')

    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    criterion = nn.L1Loss()
    with torch.no_grad():
        for mixture_mag, target_mag, _, _ in tqdm(test_loader, desc="Evaluating"):
            mixture_mag = mixture_mag.to(device)
            target_mag = target_mag.to(device)
            output, mask = model(mixture_mag)
            loss = criterion(output, target_mag)
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    BATCH_SIZE = 8
    CHUNK_DURATION = 2.0
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3

    print("Loading MUSDB18 dataset...")
    mus = musdb.DB(root="musdb18/")
    train_tracks = [t for t in mus if t.subset == 'train']
    test_tracks = [t for t in mus if t.subset == 'test']

    print(f"Train tracks: {len(train_tracks)}")
    print(f"Test tracks: {len(test_tracks)}")

    train_loader, test_loader = create_data_loaders(train_tracks, test_tracks, BATCH_SIZE, CHUNK_DURATION)

    model = SmallUNet(n_fft=2048)

    checkpoint_path = 'final_unet_model.pth'
    if os.path.isfile(checkpoint_path):
        print(f"Loading saved model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No saved model found, starting fresh training.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    train_losses, val_losses = train_model(model, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE)

    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'n_fft': 2048,
            'hop_length': 512,
            'chunk_duration': CHUNK_DURATION
        }
    }, checkpoint_path)

    print(f"Training complete! Model saved as '{checkpoint_path}'")

    test_loss = evaluate_model(model, test_loader, device)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    main()
