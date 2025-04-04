import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Initialize the dataset
root_dir = '/content/in_the_wild_dataset/release_in_the_wild'
metadata_file = '/content/modified_meta.csv'

class AudioDataset(Dataset):
    def __init__(self, root_dir, metadata_df, max_length=64600, target_sample_rate=16000):
        self.root_dir = root_dir
        self.metadata = metadata_df.reset_index(drop=True)
        self.max_length = max_length
        self.target_sample_rate = target_sample_rate
        self.resample_rate = target_sample_rate

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        label = row['label']
        folder = 'real' if label == 0 else 'fake'
        file_path = os.path.join(self.root_dir, folder, row['file'])
        
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Resample if needed
            if sample_rate != self.resample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.resample_rate
                )
                waveform = resampler(waveform)
            
            waveform = self._pad_trim(waveform)
            return waveform.unsqueeze(0), torch.tensor(label, dtype=torch.float32)
        
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            # Return zero tensor if file fails to load
            return torch.zeros(1, self.max_length), torch.tensor(0, dtype=torch.float32)

    def _pad_trim(self, waveform):
        if waveform.shape[1] > self.max_length:
            return waveform[:, :self.max_length]
        padding = self.max_length - waveform.shape[1]
        return torch.nn.functional.pad(waveform, (0, padding)) if padding > 0 else waveform

def validate_and_load_metadata(metadata_file, root_dir):
    """Load and validate metadata, ensuring files exist"""
    metadata = pd.read_csv(metadata_file)
    valid_files = []
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Validating files"):
        folder = 'real' if row['label'] == 0 else 'fake'
        file_path = os.path.join(root_dir, folder, row['file'])
        
        if os.path.exists(file_path):
            try:
                # Test loading the file
                waveform, _ = torchaudio.load(file_path)
                if waveform.shape[1] > 10:  # Minimum length check
                    valid_files.append(idx)
            except:
                continue
                
    if not valid_files:
        raise ValueError("No valid audio files found! Check your paths and data.")
    
    return metadata.loc[valid_files].copy()

# Load and validate metadata
metadata = validate_and_load_metadata(metadata_file, root_dir)

# Check class balance
print("\nClass distribution:")
print(metadata['label'].value_counts())

# Split dataset - ensuring minimum samples per class
try:
    train_metadata, val_metadata = train_test_split(
        metadata,
        test_size=0.2,
        random_state=42,
        stratify=metadata['label']
    )
except ValueError as e:
    print(f"Error in train-test split: {e}")
    print("Falling back to random split without stratification")
    train_metadata, val_metadata = train_test_split(
        metadata,
        test_size=0.2,
        random_state=42
    )

# Create datasets
train_dataset = AudioDataset(root_dir=root_dir, metadata_df=train_metadata)
val_dataset = AudioDataset(root_dir=root_dir, metadata_df=val_metadata)

print(f"\nTraining samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Create DataLoaders
def create_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(4, os.cpu_count() // 2),  # Use half of available CPUs
        pin_memory=True,
        persistent_workers=True
    )

train_loader = create_loader(train_dataset, batch_size=32, shuffle=True)
val_loader = create_loader(val_dataset, batch_size=32, shuffle=False)

# Verify we can load batches
print("\nTesting batch loading...")
try:
    test_batch = next(iter(train_loader))
    print(f"Batch loaded successfully! Shapes: {[t.shape for t in test_batch]}")
except Exception as e:
    print(f"Batch loading failed: {e}")