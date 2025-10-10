import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


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
            nn.ReLU(inplace=True),
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
            d4 = F.interpolate(
                d4, size=e4.shape[2:], mode="bilinear", align_corners=False
            )
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(
                d3, size=e3.shape[2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(
                d2, size=e2.shape[2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        mask = self.sigmoid(self.final(d1))
        output = x * mask
        return output, mask


MODEL_PATH = "final_unet_model.pth"
INPUT_SONG_PATH = "songs/song.mp3"
OUTPUT_VOCALS_PATH = "output_vocals.wav"
OUTPUT_INSTRUMENTAL_PATH = "output_instrumental.wav"

SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_DURATION = 2.0  # seconds
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

CHUNK_FRAMES = (CHUNK_SAMPLES // HOP_LENGTH) + 1


def main():
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model architecture and weights
    print(f"Loading model from {MODEL_PATH}...")
    model = SmallUNet(n_fft=N_FFT).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    # STFT and Inverse STFT transforms
    stft = T.Spectrogram(
        n_fft=N_FFT, hop_length=HOP_LENGTH, power=None, normalized=True
    ).to(device)
    istft = T.InverseSpectrogram(
        n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True
    ).to(device)

    # Load and preprocess audio
    print(f"Loading audio: {INPUT_SONG_PATH}")
    try:
        mixture_wav, sr = torchaudio.load(INPUT_SONG_PATH)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Ensure correct sample rate
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        mixture_wav = resampler(mixture_wav)

    # --- THIS IS THE FIX ---
    # Ensure audio is on the correct device
    mixture_wav = mixture_wav.to(device)

    # Handle channels: If mono, duplicate to create a stereo-like input
    if mixture_wav.shape[0] == 1:
        print("Input is mono, duplicating channel to match model's stereo input.")
        mixture_wav = mixture_wav.repeat(2, 1)
    # If more than 2 channels, take the first two
    elif mixture_wav.shape[0] > 2:
        print("Input has more than 2 channels, taking the first two.")
        mixture_wav = mixture_wav[:2, :]
    # --- END FIX ---

    # Compute STFT of the full mixture
    mixture_spec = stft(mixture_wav)
    mixture_mag = torch.abs(mixture_spec)
    mixture_phase = torch.angle(mixture_spec)

    predicted_vocals_mag = torch.zeros_like(mixture_mag)

    # Process in chunks
    print("Processing song in chunks...")
    with torch.no_grad():
        # --- THIS IS THE FIX ---

        # 1. Pad the magnitude spectrogram first
        pad_amount = CHUNK_FRAMES - (mixture_mag.shape[2] % CHUNK_FRAMES)
        mixture_mag_padded = F.pad(mixture_mag, (0, pad_amount))

        # 2. Initialize the output tensor with the padded size
        predicted_vocals_mag = torch.zeros_like(mixture_mag_padded)

        # --- END FIX ---

        for i in range(0, mixture_mag_padded.shape[2], CHUNK_FRAMES):
            chunk = mixture_mag_padded[:, :, i : i + CHUNK_FRAMES]
            chunk_max = chunk.max()
            if chunk_max > 0:  # Avoid division by zero for silent chunks
                chunk_normalized = chunk / (chunk_max + 1e-8)
                predicted_chunk_normalized, _ = model(chunk_normalized.unsqueeze(0))
                predicted_chunk = predicted_chunk_normalized * (chunk_max + 1e-8)
                predicted_vocals_mag[:, :, i : i + CHUNK_FRAMES] = (
                    predicted_chunk.squeeze(0)
                )
            else:
                predicted_vocals_mag[:, :, i : i + CHUNK_FRAMES] = 0

    # Trim the padding from the final output
    predicted_vocals_mag = predicted_vocals_mag[:, :, :-pad_amount]

    # Reconstruct audio
    print("Reconstructing and saving audio files...")
    # Vocals
    vocals_spec = predicted_vocals_mag * torch.exp(1j * mixture_phase)
    vocals_wav = istft(vocals_spec, length=mixture_wav.shape[1])

    # Instrumental (by subtracting predicted vocals from mixture)
    instrumental_wav = mixture_wav - vocals_wav

    # Save files
    torchaudio.save(OUTPUT_VOCALS_PATH, vocals_wav.cpu(), SAMPLE_RATE)
    print(f"-> Vocals saved to: {OUTPUT_VOCALS_PATH}")
    torchaudio.save(OUTPUT_INSTRUMENTAL_PATH, instrumental_wav.cpu(), SAMPLE_RATE)
    print(f"-> Instrumental saved to: {OUTPUT_INSTRUMENTAL_PATH}")
    print("Done!")


if __name__ == "__main__":
    # Create dummy song file for testing if it doesn't exist
    if not os.path.exists("songs"):
        os.makedirs("songs")
    if not os.path.exists("songs/song.mp3"):
        print("Creating a dummy 'song.mp3' for testing purposes...")
        sample_rate = 44100
        duration = 5  # seconds
        frequency = 440  # A4 note
        t = torch.linspace(
            0.0, duration, int(sample_rate * duration), dtype=torch.float32
        )
        dummy_wav = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        torchaudio.save("songs/song.mp3", dummy_wav, sample_rate)

    main()
