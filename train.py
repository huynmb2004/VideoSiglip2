import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import decord
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, get_peft_model
from model import Siglip2LateFusionBaseline

decord.bridge.set_bridge('torch')

from dataset import UCF101VideoDataset

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
    base_dir = "/media/lqngoc38/data/UCF-101/"
    annotation_dir = "/media/lqngoc38/data/UCF-101/annotations/ucfTrainTestlist/"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "google/siglip2-base-patch16-224"
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Setting up Datasets and DataLoaders...")
    train_dataset = UCF101VideoDataset(base_dir=base_dir, annotation_dir=annotation_dir, processor=processor, num_frames=8, mode='train')
    val_dataset = UCF101VideoDataset(base_dir=base_dir, annotation_dir=annotation_dir, processor=processor, num_frames=8, mode='val')

    # Safe to use num_workers=4 since we are inside __main__, but lowering to 0 to prevent OOM
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    
    print("Initializing Baseline Model...")
    model = Siglip2LateFusionBaseline(model_name=model_name, num_classes=train_dataset._get_num_classes()).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        model.train()
        train_running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)
            
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            
            train_running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100 * correct / total:.2f}%"})
        train_acc = 100 * correct / total
        train_loss = train_running_loss / len(train_loader)

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for batch in progress_bar:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label_id"].to(device)
            
                logits = model(pixel_values)
                loss = criterion(logits, labels)
            
                val_running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100 * correct / total:.2f}%"})
        val_acc = 100 * correct / total
        val_loss = val_running_loss / len(val_loader)

        print(f"Summary -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
