import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import decord
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, get_peft_model

decord.bridge.set_bridge('torch')

class UCF101VideoDataset(Dataset):
    """
    Custom Dataset for UCF101 Video Action Recognition.
    Reads videos using decord and generates inputs for SigLIP.
    """
    def __init__(self, csv_file, base_dir, processor, num_frames=8, mode='train', subset_classes=None):
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.processor = processor
        self.num_frames = num_frames
        self.mode = mode
        
        # CRITICAL: Limit to N classes for rapid testing
        if subset_classes is not None:
            unique_classes = sorted(self.data['label'].unique().tolist())
            target_classes = unique_classes[:subset_classes]
            self.data = self.data[self.data['label'].isin(target_classes)].reset_index(drop=True)
            
        self.unique_labels = sorted(self.data['label'].unique().tolist())
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        
    def __len__(self):
        return len(self.data)
        
    def _get_frame_indices(self, total_frames):
        if total_frames < self.num_frames:
            # Pad by linearly spacing if video is too short
            return np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        seg_size = total_frames / self.num_frames
        indices = []
        for i in range(self.num_frames):
            start = int(i * seg_size)
            end = int((i + 1) * seg_size)
            if self.mode == 'train':
                # Random sampling in the segment
                idx = np.random.randint(start, end)
            else:
                # Center sampling in the segment
                idx = start + (end - start) // 2
            indices.append(idx)
        return np.array(indices)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        clip_path = row['clip_path']
        label_str = row['label']
        label_id = self.label_to_id[label_str]
        
        if clip_path.startswith('/') or clip_path.startswith('\\'):
            clip_path = clip_path[1:]
        video_path = os.path.join(self.base_dir, clip_path)
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = self._get_frame_indices(total_frames)
            frames = vr.get_batch(frame_indices) # Shape: (T, H, W, C)
        except Exception as e:
            # print(f"Error reading video {video_path}: {e}")
            frames = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.uint8).permute(0, 2, 3, 1)

        frames_np = [frame.numpy() for frame in frames]
        
        # Apply Processor (Handles Resizing, Normalization)
        inputs = self.processor(
            images=frames_np, 
            return_tensors="pt"
        )
        
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0) # (T, C, H, W)
            
        return {
            "pixel_values": pixel_values,
            "label_id": torch.tensor(label_id, dtype=torch.long)
        }

class Siglip2LateFusionBaseline(nn.Module):
    """
    Baseline Video Action Recognition model using SigLIP-2 with Late Fusion (Mean Pooling) and LoRA.
    """
    def __init__(self, model_name="google/siglip2-base-patch16-224", num_classes=5):
        super().__init__()
        # Load Pretrained SigLIP-2 Base Model
        self.base_model = AutoModel.from_pretrained(model_name)
        self.vision_encoder = self.base_model.vision_model
        
        # 1. ABSOLUTELY DO NOT UNFREEZE THE ENTIRE MODEL
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # 2. Inject LoRA adapters via peft
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        
        self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
        print("--- LoRA Applied to Vision Encoder ---")
        self.vision_encoder.print_trainable_parameters()
            
        # Extract hidden size safely from the original wrapped config
        hidden_size = self.base_model.vision_model.config.hidden_size
        
        # 3. Linear Classification Head
        self.classifier = nn.Linear(hidden_size, num_classes)
        # Ensure classifier has gradients
        for param in self.classifier.parameters():
            param.requires_grad = True
        
    def forward(self, pixel_values):
        # pixel_values shape: (B, T, C, H, W)
        B, T, C, H, W = pixel_values.shape
        
        # Flatten B and T to pass through 2D vision encoder
        pixel_values = pixel_values.view(B * T, C, H, W)
        
        # Extract frame features
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        frame_features = vision_outputs.pooler_output # (B*T, D)
        
        # Late Fusion (Mean Pooling)
        frame_features = frame_features.view(B, T, -1) # (B, T, D)
        video_features = frame_features.mean(dim=1) # (B, D)
        
        # Classification
        logits = self.classifier(video_features) # (B, num_classes)
        return logits

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label_id"].to(device)
        
        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100 * correct / total:.2f}%"})
        
    return running_loss / len(dataloader), 100 * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)
            
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100 * correct / total:.2f}%"})
            
    return running_loss / len(dataloader), 100 * correct / total

def main():
    base_dir = r"C:\Users\Admin\.cache\kagglehub\datasets\matthewjansen\ucf101-action-recognition\versions\4"
    train_csv = os.path.join(base_dir, "train.csv")
    val_csv = os.path.join(base_dir, "val.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "google/siglip2-base-patch16-224"
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Setting up Datasets and DataLoaders...")
    subset_classes = 5
    train_dataset = UCF101VideoDataset(train_csv, base_dir, processor, num_frames=8, mode='train', subset_classes=subset_classes)
    val_dataset = UCF101VideoDataset(val_csv, base_dir, processor, num_frames=8, mode='val', subset_classes=subset_classes)

    # Safe to use num_workers=4 since we are inside __main__
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    
    print("Initializing Baseline Model...")
    model = Siglip2LateFusionBaseline(model_name=model_name, num_classes=subset_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Summary -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
