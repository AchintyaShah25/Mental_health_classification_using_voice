"""
Wav2Vec2-based Depression Detection Model
Uses pretrained Wav2Vec2 embeddings with attention-based classifier
Optimized for M4 MacBook Air
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
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_auc_score
)
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
    OUTPUT_DIR = "./output_wav2vec"
    
    # Audio
    SAMPLE_RATE = 16000
    MAX_DURATION = 15  # seconds
    
    # Training
    BATCH_SIZE = 4  # Smaller batch for transformer
    EPOCHS = 5
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-2
    
    # Model
    WAV2VEC_MODEL = "facebook/wav2vec2-base"  # Smaller, faster
    FREEZE_ENCODER = True  # Freeze Wav2Vec2 weights
    HIDDEN_DIM = 256
    NUM_CLASSES = 2
    
    # Device
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(Config.SEED)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"Using device: {Config.DEVICE}")

# ============== DATASET ==============
class Wav2VecDataset(Dataset):
    def __init__(self, df, audio_dir, processor, max_length, augment=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.df)
    
    def load_audio(self, participant_id):
        audio_path = self.audio_dir / f"{participant_id}_AUDIO.wav"
        if not audio_path.exists():
            audio_path = self.audio_dir / f"{participant_id}.wav"
        
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != Config.SAMPLE_RATE:
            resampler = T.Resample(sr, Config.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform.squeeze(0)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['Participant_ID']
        label = row['PHQ_Binary']
        
        try:
            waveform = self.load_audio(pid)
        except:
            waveform = torch.zeros(self.max_length)
        
        # Truncate or pad
        if len(waveform) > self.max_length:
            # Random crop during training
            if self.augment:
                start = random.randint(0, len(waveform) - self.max_length)
            else:
                start = 0
            waveform = waveform[start:start + self.max_length]
        else:
            padding = self.max_length - len(waveform)
            waveform = F.pad(waveform, (0, padding))
        
        # Add noise augmentation
        if self.augment and random.random() > 0.5:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        return waveform, torch.tensor(label, dtype=torch.long), pid

def collate_fn(batch):
    waveforms, labels, pids = zip(*batch)
    waveforms = torch.stack(waveforms)
    labels = torch.stack(labels)
    return waveforms, labels, pids

# ============== MODEL ==============
class AttentionHead(nn.Module):
    """Multi-head self-attention for sequence pooling"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.attention_weights = None
    
    def forward(self, x):
        B, T, D = x.shape
        
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        self.attention_weights = attn.mean(dim=1)  # Average across heads
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        
        # Global pooling with attention
        pooled = (out * self.attention_weights.mean(dim=1, keepdim=True).transpose(1, 2)).sum(dim=1)
        return pooled

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes=2, freeze_encoder=True):
        super().__init__()
        
        # Load pretrained Wav2Vec2
        self.wav2vec = Wav2Vec2Model.from_pretrained(Config.WAV2VEC_MODEL)
        
        if freeze_encoder:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
            # Unfreeze last 2 layers for fine-tuning
            for param in self.wav2vec.encoder.layers[-2:].parameters():
                param.requires_grad = True
        
        hidden_size = self.wav2vec.config.hidden_size  # 768 for base
        
        # Projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Attention pooling
        self.attention = AttentionHead(Config.HIDDEN_DIM, num_heads=4)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(Config.HIDDEN_DIM // 2, num_classes)
        )
    
    def forward(self, x, attention_mask=None):
        # x: (batch, seq_len)
        outputs = self.wav2vec(x, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, time, 768)
        
        # Project
        hidden = self.proj(hidden)
        
        # Attention pooling
        pooled = self.attention(hidden)
        
        # Classify
        logits = self.classifier(pooled)
        return logits
    
    def get_attention_weights(self):
        return self.attention.attention_weights

# ============== TRAINING ==============
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for waveforms, labels, _ in tqdm(loader, desc="Training"):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(loader), acc, f1

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for waveforms, labels, _ in tqdm(loader, desc="Validating"):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return total_loss / len(loader), acc, f1, mcc, auc, all_preds, all_labels, all_probs

def visualize_attention_wav2vec(model, dataset, device, save_path, num_samples=4):
    """Visualize attention weights over audio"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3*num_samples))
    
    # Get diverse samples
    depressed_idx = [i for i in range(len(dataset)) if dataset.df.iloc[i]['PHQ_Binary'] == 1]
    healthy_idx = [i for i in range(len(dataset)) if dataset.df.iloc[i]['PHQ_Binary'] == 0]
    
    samples = []
    if depressed_idx:
        samples.extend(random.sample(depressed_idx, min(num_samples//2, len(depressed_idx))))
    if healthy_idx:
        samples.extend(random.sample(healthy_idx, min(num_samples//2, len(healthy_idx))))
    
    for i, idx in enumerate(samples[:num_samples]):
        waveform, label, pid = dataset[idx]
        waveform = waveform.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(waveform)
            pred = output.argmax(dim=1).item()
            prob = F.softmax(output, dim=1)[0, pred].item()
            attention = model.get_attention_weights()[0].mean(dim=0).cpu().numpy()
        
        wave_np = waveform.squeeze().cpu().numpy()
        
        # Plot waveform
        time_axis = np.linspace(0, len(wave_np)/Config.SAMPLE_RATE, len(wave_np))
        axes[i, 0].plot(time_axis, wave_np, alpha=0.7, linewidth=0.5)
        axes[i, 0].set_title(f'Participant {pid} | True: {"Depressed" if label else "Healthy"}')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        
        # Plot attention
        attn_time = np.linspace(0, len(wave_np)/Config.SAMPLE_RATE, len(attention))
        axes[i, 1].fill_between(attn_time, attention, alpha=0.6, color='coral')
        axes[i, 1].plot(attn_time, attention, color='red', linewidth=1)
        axes[i, 1].set_title(f'Attention | Pred: {"Depressed" if pred else "Healthy"} ({prob:.1%})')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('Attention Weight')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention visualization to {save_path}")

# ============== MAIN ==============
def main():
    print("\n" + "="*60)
    print("Wav2Vec2 Depression Detection Training")
    print("="*60)
    
    # Load labels
    train_df = pd.read_csv(os.path.join(Config.LABELS_DIR, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(Config.LABELS_DIR, "dev_split.csv"))
    test_df = pd.read_csv(os.path.join(Config.LABELS_DIR, "test_split.csv"))
    
    # Check available files
    audio_dir = Path(Config.AUDIO_DIR)
    found = [int(f.stem.split('_')[0]) for f in audio_dir.glob("*_AUDIO.wav")]
    
    train_df = train_df[train_df['Participant_ID'].isin(found)]
    val_df = val_df[val_df['Participant_ID'].isin(found)]
    test_df = test_df[test_df['Participant_ID'].isin(found)]
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained(Config.WAV2VEC_MODEL)
    max_length = Config.SAMPLE_RATE * Config.MAX_DURATION
    
    # Datasets
    train_ds = Wav2VecDataset(train_df, Config.AUDIO_DIR, processor, max_length, augment=True)
    val_ds = Wav2VecDataset(val_df, Config.AUDIO_DIR, processor, max_length, augment=False)
    test_ds = Wav2VecDataset(test_df, Config.AUDIO_DIR, processor, max_length, augment=False)
    
    # Weighted sampler
    labels = train_df['PHQ_Binary'].tolist()
    counts = Counter(labels)
    weights = [1.0/counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_ds, Config.BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = Wav2Vec2Classifier(Config.NUM_CLASSES, Config.FREEZE_ENCODER).to(Config.DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    
    # Loss with class weights
    class_weights = torch.tensor([
        len(labels)/(2*counts[0]),
        len(labels)/(2*counts[1])
    ]).to(Config.DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Config.EPOCHS)
    
    # Training loop
    best_f1 = 0
    history = {'train_loss':[], 'val_loss':[], 'val_f1':[], 'val_mcc':[]}
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc, val_f1, val_mcc, val_auc, _, _, _ = validate(model, val_loader, criterion, Config.DEVICE)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_mcc'].append(val_mcc)
        
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | F1: {val_f1:.4f} | MCC: {val_mcc:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {'max_length': max_length, 'wav2vec_model': Config.WAV2VEC_MODEL}
            }, os.path.join(Config.OUTPUT_DIR, 'best_wav2vec_model.pt'))
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Test Evaluation")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(Config.OUTPUT_DIR, 'best_wav2vec_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_acc, test_f1, test_mcc, test_auc, preds, labels, probs = validate(
        model, test_loader, criterion, Config.DEVICE
    )
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test MCC: {test_mcc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Healthy', 'Depressed']))
    
    # Visualizations
    visualize_attention_wav2vec(model, test_ds, Config.DEVICE, 
                                os.path.join(Config.OUTPUT_DIR, 'wav2vec_attention.png'))
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Depressed'],
                yticklabels=['Healthy', 'Depressed'])
    plt.title('Wav2Vec2 - Confusion Matrix')
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'wav2vec_confusion.png'), dpi=150)
    plt.close()
    
    print(f"\nResults saved to {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()