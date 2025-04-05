import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict

# Import AASIST model
from aasist.models import AASISTModel

# Dynamic Chunk Size dataset implementation
class DynamicChunkAudioDataset(Dataset):
    def __init__(self, root_dir, metadata_df, min_length=16000, max_length=64600, target_sample_rate=16000):
        self.root_dir = root_dir
        self.metadata = metadata_df.reset_index(drop=True)
        self.min_length = min_length
        self.max_length = max_length
        self.target_sample_rate = target_sample_rate
        
        # Get audio durations for ALMFT
        self.durations = {}
        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Loading durations"):
            folder = 'real' if row['label'] == 0 else 'fake'
            file_path = os.path.join(self.root_dir, folder, row['file'])
            try:
                waveform, _ = torchaudio.load(file_path)
                self.durations[idx] = waveform.shape[1]
            except:
                self.durations[idx] = 0

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        label = row['label']
        folder = 'real' if row['label'] == 0 else 'fake'
        file_path = os.path.join(self.root_dir, folder, row['file'])
        
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
            
            # Store original duration for ALMFT
            duration = waveform.shape[1]
            
            # DCS - Randomly select chunk size for this sample
            if waveform.shape[1] > self.min_length:
                # Choose a random length between min_length and the actual length (or max_length)
                chunk_size = random.randint(self.min_length, min(waveform.shape[1], self.max_length))
                
                # If original is longer than chosen chunk size, randomly select starting point
                if waveform.shape[1] > chunk_size:
                    start = random.randint(0, waveform.shape[1] - chunk_size)
                    waveform = waveform[:, start:start+chunk_size]
            
            # Always pad/trim to max_length for batch processing
            waveform = self._pad_trim(waveform)

            return waveform.unsqueeze(0), torch.tensor(label, dtype=torch.float32), torch.tensor(duration, dtype=torch.float32)
        
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return torch.zeros(1, self.max_length), torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32)

    def _pad_trim(self, waveform):
        if waveform.shape[1] > self.max_length:
            return waveform[:, :self.max_length]
        padding = self.max_length - waveform.shape[1]
        return torch.nn.functional.pad(waveform, (0, padding)) if padding > 0 else waveform


# Loss functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class AdaptiveLargeMarginLoss(nn.Module):
    def __init__(self, base_loss, min_margin=0.2, max_margin=0.9, duration_weight=0.3):
        super(AdaptiveLargeMarginLoss, self).__init__()
        self.base_loss = base_loss
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.duration_weight = duration_weight
        
    def forward(self, inputs, targets, durations=None):
        if durations is None:
            return self.base_loss(inputs, targets)
        
        # Normalize durations to 0-1 range for adaptive margin
        # Shorter utterances get larger margins for more robust learning
        max_duration = torch.max(durations).item()
        if max_duration > 0:
            norm_durations = durations / max_duration
            # Inverse relationship: shorter duration = larger margin
            adaptive_margins = self.max_margin - (norm_durations * (self.max_margin - self.min_margin))
        else:
            adaptive_margins = torch.ones_like(targets) * self.min_margin
        
        # Apply margins to predictions before computing loss
        adjusted_inputs = inputs * (1 + adaptive_margins * (2*targets - 1))
        
        # Calculate base loss with adjusted inputs
        base_loss = self.base_loss(adjusted_inputs, targets)
        
        # Add duration-aware regularization term
        duration_penalty = self.duration_weight * torch.mean(1.0 / (1.0 + durations/1000))
        
        return base_loss + duration_penalty


# Sharpness-Aware Minimization (SAM) optimizer
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        
        # First step computes gradient at w + e(w)
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        
        # Second step updates parameters with sharpness-aware gradient
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# Enhanced training function with SAM and ALMFT
def train(model, dataloader, criterion, optimizer, device, is_sam=True):
    model.train()
    running_loss = 0.0
    duration_stats = defaultdict(list)
    
    for inputs, labels, durations in tqdm(dataloader, desc="Training"):
        inputs, labels, durations = inputs.to(device), labels.to(device).unsqueeze(1), durations.to(device)
        
        if is_sam:
            # SAM optimizer requires closure
            def closure():
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze(), durations)
                loss.backward()
                return loss
            
            optimizer.zero_grad()
            loss = closure()
            optimizer.step(closure)
            running_loss += loss.item()
        else:
            # Standard optimizer path
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze(), durations)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Track duration statistics for analysis
        for dur in durations.cpu().numpy():
            bin_key = int(dur // 1000)  # bin by seconds
            duration_stats[bin_key].append(1)
    
    # Print duration distribution
    print("Sample duration distribution:")
    for k in sorted(duration_stats.keys()):
        print(f"{k}s - {k+1}s: {len(duration_stats[k])} samples")
    
    return running_loss / len(dataloader)


# Enhanced validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    all_preds = []
    all_labels = []
    duration_accuracy = defaultdict(lambda: [0, 0])  # correct, total for each duration bin
    
    with torch.no_grad():
        for inputs, labels, durations in tqdm(dataloader, desc="Validation"):
            inputs, labels, durations = inputs.to(device), labels.to(device).unsqueeze(1), durations.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs.squeeze(), labels.squeeze(), durations)
            
            running_loss += loss.item()
            
            predictions = (outputs > 0.5).float()
            
            # Track regular accuracy
            correct_batch = (predictions == labels).sum().item()
            correct_predictions += correct_batch
            
            # Track duration-based accuracy
            for i, dur in enumerate(durations.cpu().numpy()):
                bin_key = int(dur // 1000)  # bin by seconds
                is_correct = (predictions[i] == labels[i]).item()
                duration_accuracy[bin_key][0] += int(is_correct)
                duration_accuracy[bin_key][1] += 1
            
            # Store predictions and labels for detailed metrics
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate overall accuracy
    accuracy = correct_predictions / len(dataloader.dataset)
    
    # Print duration-based accuracy
    print("\nAccuracy by duration:")
    for k in sorted(duration_accuracy.keys()):
        if duration_accuracy[k][1] > 0:  # Avoid division by zero
            dur_acc = duration_accuracy[k][0] / duration_accuracy[k][1]
            print(f"{k}s - {k+1}s: {dur_acc:.4f} ({duration_accuracy[k][0]}/{duration_accuracy[k][1]})")
    
    return running_loss / len(dataloader), accuracy


# Main script for training and validation
if __name__ == "__main__":
    root_dir = '/content/in_the_wild_dataset/release_in_the_wild'
    metadata_file = '/content/modified_meta.csv'

    # Load and validate metadata
    metadata_df = validate_and_load_metadata(metadata_file=metadata_file, root_dir=root_dir)

    # Split dataset into training and validation sets
    train_metadata_df, val_metadata_df = train_test_split(
        metadata_df,
        test_size=0.2,
        random_state=42,
        stratify=metadata_df['label']
    )

    # Create datasets with Dynamic Chunk Size (DCS)
    train_dataset = DynamicChunkAudioDataset(
        root_dir=root_dir, 
        metadata_df=train_metadata_df,
        min_length=16000,  # 1 second minimum
        max_length=64600   # ~4 seconds maximum
    )
    
    val_dataset = DynamicChunkAudioDataset(
        root_dir=root_dir, 
        metadata_df=val_metadata_df,
        min_length=16000,
        max_length=64600
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,       # Smaller batch size for SAM
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True      # Important for SAM to have consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AASISTModel().to(device)
    
    # Initialize base loss function - focal loss with BCE
    base_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Wrap with Adaptive Large Margin
    criterion = AdaptiveLargeMarginLoss(
        base_loss=base_criterion,
        min_margin=0.2,
        max_margin=0.9,
        duration_weight=0.3
    )
    
    # Initialize SAM optimizer with Adam base
    base_optimizer = torch.optim.Adam
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        lr=0.001,
        rho=0.05,        # SAM hyperparameter
        adaptive=True,   # Use ASAM instead of SAM
        weight_decay=1e-5
    )
    
    # Initialize cosine annealing scheduler
    scheduler = CosineAnnealingLR(
        optimizer.base_optimizer,  # Important: schedule the base optimizer
        T_max=30,                 # Max epochs
        eta_min=1e-6              # Minimum learning rate
    )

    num_epochs = 30
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        current_lr = optimizer.base_optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Training
        train_loss = train(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_sam=True  # Use SAM optimization
        )

        # Validation
        val_loss, val_accuracy = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Update learning rate
        scheduler.step()
        
        # Save the best model weights based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
            }, "best_aasist_model.pth")
            print("Saved best model!")
        
        # Save intermediate model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
            }, f"aasist_model_epoch_{epoch+1}.pth")
            
    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
return best_val_accuracy
