"""
Depression Detection from Audio using CNN + Transformer
Optimized for M4 MacBook Air with MPS acceleration
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from collections import Counter
import random
import json
from datetime import datetime

# ============== CONFIGURATION ==============
class Config:
    # Paths - UPDATE THESE
    AUDIO_DIR = "/Volumes/One Touch/Fai proj/DATA/collected wav"
    LABELS_DIR = '/Volumes/One Touch/Fai proj/DATA/labels'
    OUTPUT_DIR = "./output"
    
    # Audio processing
    SAMPLE_RATE = 16000
    DURATION = 10  # seconds - use 10s chunks
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    
    # Training
    BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Model
    MODEL_TYPE = "transformer"  # "cnn" or "transformer"
    NUM_CLASSES = 2
    
    # Device
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Reproducibility
    SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(Config.SEED)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

print(f"Using device: {Config.DEVICE}")

# ============== DATA LOADING ==============
def load_labels():
    """Load and combine train/dev/test splits"""
    train_df = pd.read_csv(os.path.join(Config.LABELS_DIR, "train_split.csv"))
    dev_df = pd.read_csv(os.path.join(Config.LABELS_DIR, "dev_split.csv"))
    test_df = pd.read_csv(os.path.join(Config.LABELS_DIR, "test_split.csv"))
    
    train_df['split'] = 'train'
    dev_df['split'] = 'dev'
    test_df['split'] = 'test'
    
    df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    print(f"\n{'='*50}")
    print("Dataset Distribution:")
    print(f"{'='*50}")
    for split in ['train', 'dev', 'test']:
        split_df = df[df['split'] == split]
        dep = (split_df['PHQ_Binary'] == 1).sum()
        total = len(split_df)
        print(f"{split.upper():6s}: {total:3d} samples | Depressed: {dep:2d} ({dep/total*100:.1f}%)")
    
    return df

# ============== AUDIO AUGMENTATION ==============
class AudioAugmentation:
    """Audio augmentations for training"""
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        
    def time_stretch(self, waveform, rate=None):
        if rate is None:
            rate = random.uniform(0.9, 1.1)
        effects = [["tempo", str(rate)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sr, effects, channels_first=True
        )
        return augmented
    
    def pitch_shift(self, waveform, n_steps=None):
        if n_steps is None:
            n_steps = random.uniform(-2, 2)
        effects = [["pitch", str(n_steps * 100)], ["rate", str(self.sr)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sr, effects, channels_first=True
        )
        return augmented
    
    def add_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def time_mask(self, spec, max_mask=30):
        """Mask random time steps in spectrogram"""
        cloned = spec.clone()
        t = spec.shape[-1]
        t_mask = random.randint(0, max_mask)
        t_start = random.randint(0, max(0, t - t_mask))
        cloned[..., t_start:t_start + t_mask] = 0
        return cloned
    
    def freq_mask(self, spec, max_mask=20):
        """Mask random frequency bands in spectrogram"""
        cloned = spec.clone()
        f = spec.shape[-2]
        f_mask = random.randint(0, max_mask)
        f_start = random.randint(0, max(0, f - f_mask))
        cloned[..., f_start:f_start + f_mask, :] = 0
        return cloned

# ============== DATASET ==============
class DepressionAudioDataset(Dataset):
    def __init__(self, df, audio_dir, transform=None, augment=False, num_chunks=3):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.augment = augment
        self.num_chunks = num_chunks  # Number of random chunks per audio
        self.aug = AudioAugmentation(Config.SAMPLE_RATE) if augment else None
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS
        )
        self.db_transform = T.AmplitudeToDB()
        
    def __len__(self):
        return len(self.df)
    
    def load_audio(self, participant_id):
        """Load audio file for a participant"""
        audio_path = self.audio_dir / f"{participant_id}_AUDIO.wav"
        
        if not audio_path.exists():
            # Try alternative naming
            audio_path = self.audio_dir / f"{participant_id}.wav"
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != Config.SAMPLE_RATE:
            resampler = T.Resample(sr, Config.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform
    
    def extract_chunk(self, waveform, chunk_samples):
        """Extract a random chunk from waveform"""
        total_samples = waveform.shape[1]
        
        if total_samples <= chunk_samples:
            # Pad if too short
            padding = chunk_samples - total_samples
            waveform = F.pad(waveform, (0, padding))
        else:
            # Random crop
            start = random.randint(0, total_samples - chunk_samples)
            waveform = waveform[:, start:start + chunk_samples]
        
        return waveform
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        participant_id = row['Participant_ID']
        label = row['PHQ_Binary']
        
        try:
            waveform = self.load_audio(participant_id)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            # Return zeros if file not found
            chunk_samples = Config.SAMPLE_RATE * Config.DURATION
            waveform = torch.zeros(1, chunk_samples)
        
        # Extract chunk
        chunk_samples = Config.SAMPLE_RATE * Config.DURATION
        waveform = self.extract_chunk(waveform, chunk_samples)
        
        # Apply augmentations
        if self.augment and self.aug:
            if random.random() > 0.5:
                waveform = self.aug.add_noise(waveform)
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Apply spectrogram augmentations
        if self.augment and self.aug:
            if random.random() > 0.5:
                mel_spec = self.aug.time_mask(mel_spec)
            if random.random() > 0.5:
                mel_spec = self.aug.freq_mask(mel_spec)
        
        return mel_spec, torch.tensor(label, dtype=torch.long), participant_id

# ============== MODELS ==============
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class CNNModel(nn.Module):
    """Lightweight CNN for spectrogram classification"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # For Grad-CAM
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        x = self.features(x)
        
        # Register hook for Grad-CAM
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x
        
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations

class AttentionPool(nn.Module):
    """Attention pooling for transformer features"""
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, dim)
        weights = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(weights, dim=1)
        self.attention_weights = weights.squeeze(-1)  # Save for visualization
        pooled = (x * weights).sum(dim=1)  # (batch, dim)
        return pooled

class TransformerModel(nn.Module):
    """Transformer-based model with attention visualization"""
    def __init__(self, input_dim=128, num_classes=2, num_heads=2, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, 256)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attention_pool = AttentionPool(256)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        x = x.squeeze(1)  # (batch, n_mels, time)
        x = x.permute(0, 2, 1)  # (batch, time, n_mels)
        
        x = self.input_proj(x)  # (batch, time, 256)
        x = self.transformer(x)
        x = self.attention_pool(x)  # (batch, 256)
        x = self.classifier(x)
        return x
    
    def get_attention_weights(self):
        return self.attention_pool.attention_weights

# ============== TRAINING ==============
def get_weighted_sampler(labels):
    """Create weighted sampler for imbalanced dataset"""
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="Training")
    for specs, labels, _ in pbar:
        specs = specs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, acc, f1

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for specs, labels, _ in tqdm(loader, desc="Validating"):
            specs = specs.to(device)
            labels = labels.to(device)
            
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return avg_loss, acc, f1, mcc, auc, all_preds, all_labels, all_probs

# ============== VISUALIZATION ==============
def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train')
    axes[1, 0].plot(history['val_f1'], label='Validation')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    
    # MCC
    axes[1, 1].plot(history['val_mcc'], label='Validation MCC')
    axes[1, 1].plot(history['val_auc'], label='Validation AUC')
    axes[1, 1].set_title('MCC & AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Depressed'],
                yticklabels=['Healthy', 'Depressed'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def compute_gradcam(model, input_tensor, target_class):
    """Compute Grad-CAM for CNN model"""
    model.eval()
    input_tensor.requires_grad_(True)
    
    output = model(input_tensor)
    
    model.zero_grad()
    target = output[0, target_class]
    target.backward()
    
    gradients = model.get_activations_gradient()
    activations = model.get_activations()
    
    # Pool gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Weight activations
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    
    return heatmap.detach().cpu().numpy()

def visualize_gradcam(model, dataset, device, save_path, num_samples=4):
    """Visualize Grad-CAM for sample predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    
    # Get samples (try to get both classes)
    depressed_idx = [i for i in range(len(dataset)) if dataset.df.iloc[i]['PHQ_Binary'] == 1]
    healthy_idx = [i for i in range(len(dataset)) if dataset.df.iloc[i]['PHQ_Binary'] == 0]
    
    sample_indices = []
    if len(depressed_idx) >= num_samples // 2:
        sample_indices.extend(random.sample(depressed_idx, num_samples // 2))
    if len(healthy_idx) >= num_samples // 2:
        sample_indices.extend(random.sample(healthy_idx, num_samples // 2))
    
    for i, idx in enumerate(sample_indices[:num_samples]):
        spec, label, pid = dataset[idx]
        spec = spec.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(spec)
            pred = output.argmax(dim=1).item()
            prob = F.softmax(output, dim=1)[0, pred].item()
        
        # Compute Grad-CAM
        spec.requires_grad_(True)
        heatmap = compute_gradcam(model, spec, pred)
        
        spec_np = spec.squeeze().cpu().detach().numpy()
        
        # Plot original spectrogram
        axes[i, 0].imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'Participant {pid}\nTrue: {"Depressed" if label else "Healthy"}')
        axes[i, 0].set_ylabel('Mel Frequency')
        
        # Plot Grad-CAM heatmap
        heatmap_resized = np.array(
            plt.cm.jet(
                np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 1))
            )[:, :, :3]
        )
        # Resize heatmap to match spectrogram
        from scipy.ndimage import zoom
        zoom_factors = (spec_np.shape[0] / heatmap.shape[0], 
                       spec_np.shape[1] / heatmap.shape[1])
        heatmap_resized = zoom(heatmap, zoom_factors, order=1)
        
        axes[i, 1].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet')
        axes[i, 1].set_title(f'Grad-CAM Attention')
        
        # Plot overlay
        axes[i, 2].imshow(spec_np, aspect='auto', origin='lower', cmap='viridis', alpha=0.7)
        axes[i, 2].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.4)
        axes[i, 2].set_title(f'Pred: {"Depressed" if pred else "Healthy"} ({prob:.2%})')
        axes[i, 2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM visualization saved to {save_path}")

def visualize_attention(model, dataset, device, save_path, num_samples=4):
    """Visualize attention weights for transformer model"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))
    
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(sample_indices):
        spec, label, pid = dataset[idx]
        spec = spec.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(spec)
            pred = output.argmax(dim=1).item()
            prob = F.softmax(output, dim=1)[0, pred].item()
            attention = model.get_attention_weights().cpu().numpy()[0]
        
        spec_np = spec.squeeze().cpu().numpy()
        
        # Plot spectrogram
        axes[i, 0].imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'Participant {pid} | True: {"Depressed" if label else "Healthy"}')
        axes[i, 0].set_ylabel('Mel Frequency')
        axes[i, 0].set_xlabel('Time Frame')
        
        # Plot attention weights
        axes[i, 1].bar(range(len(attention)), attention, color='coral')
        axes[i, 1].set_title(f'Attention Weights | Pred: {"Depressed" if pred else "Healthy"} ({prob:.2%})')
        axes[i, 1].set_xlabel('Time Frame')
        axes[i, 1].set_ylabel('Attention Weight')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention visualization saved to {save_path}")

# ============== MAIN TRAINING ==============
def main():
    print("\n" + "="*60)
    print("Depression Detection Model Training")
    print("="*60)
    
    # Load data
    df = load_labels()
    
    # Verify audio files exist
    audio_dir = Path(Config.AUDIO_DIR)
    found_files = list(audio_dir.glob("*_AUDIO.wav"))
    print(f"\nFound {len(found_files)} audio files in {audio_dir}")
    
    if len(found_files) == 0:
        print("WARNING: No audio files found! Check your AUDIO_DIR path.")
        return
    
    # Filter df to only include participants with audio
    available_ids = [int(f.stem.split('_')[0]) for f in found_files]
    df = df[df['Participant_ID'].isin(available_ids)]
    print(f"Matched {len(df)} participants with audio files")
    
    # Split data
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'dev']
    test_df = df[df['split'] == 'test']
    
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = DepressionAudioDataset(train_df, Config.AUDIO_DIR, augment=True)
    val_dataset = DepressionAudioDataset(val_df, Config.AUDIO_DIR, augment=False)
    test_dataset = DepressionAudioDataset(test_df, Config.AUDIO_DIR, augment=False)
    
    # Create weighted sampler for training
    train_labels = train_df['PHQ_Binary'].tolist()
    sampler = get_weighted_sampler(train_labels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, 
        sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, 
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, 
        shuffle=False, num_workers=0
    )
    
    # Initialize model
    if Config.MODEL_TYPE == "cnn":
        model = CNNModel(num_classes=Config.NUM_CLASSES)
    else:
        model = TransformerModel(input_dim=Config.N_MELS, num_classes=Config.NUM_CLASSES)
    
    model = model.to(Config.DEVICE)
    print(f"\nModel: {Config.MODEL_TYPE.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights for focal loss
    class_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor([
        total / (2 * class_counts[0]),  # Healthy
        total / (2 * class_counts[1])   # Depressed
    ]).to(Config.DEVICE)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_mcc': [], 'val_auc': []
    }
    
    best_f1 = 0
    best_model_path = os.path.join(Config.OUTPUT_DIR, 'best_model.pt')
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_mcc, val_auc, _, _, _ = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_mcc'].append(val_mcc)
        history['val_auc'].append(val_auc)
        
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | MCC: {val_mcc:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': {
                    'model_type': Config.MODEL_TYPE,
                    'n_mels': Config.N_MELS,
                    'sample_rate': Config.SAMPLE_RATE,
                    'duration': Config.DURATION
                }
            }, best_model_path)
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
    
    # ============== EVALUATION ==============
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_acc, test_f1, test_mcc, test_auc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, Config.DEVICE
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  MCC:      {test_mcc:.4f}")
    print(f"  AUC:      {test_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        test_labels, test_preds, 
        target_names=['Healthy', 'Depressed'],
        zero_division=0
    ))
    
    # Save results
    results = {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_mcc': test_mcc,
        'test_auc': test_auc,
        'best_val_f1': best_f1,
        'model_type': Config.MODEL_TYPE,
        'epochs': Config.EPOCHS,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # ============== VISUALIZATIONS ==============
    print("\nGenerating visualizations...")
    
    # Training history
    plot_training_history(history, os.path.join(Config.OUTPUT_DIR, 'training_history.png'))
    
    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds, os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'))
    
    # Model interpretability
    if Config.MODEL_TYPE == "cnn":
        visualize_gradcam(model, test_dataset, Config.DEVICE, 
                         os.path.join(Config.OUTPUT_DIR, 'gradcam_visualization.png'))
    else:
        visualize_attention(model, test_dataset, Config.DEVICE,
                           os.path.join(Config.OUTPUT_DIR, 'attention_visualization.png'))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Model saved to: {best_model_path}")
    print(f"Results saved to: {Config.OUTPUT_DIR}")
    print("="*60)
    
    return model, history, results

if __name__ == "__main__":
    model, history, results = main()
