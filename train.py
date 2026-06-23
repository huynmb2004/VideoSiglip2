import os
import argparse
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

def main(args):
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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    
    print("Initializing Baseline Model...")
    model = Siglip2LateFusionBaseline(model_name=model_name, num_classes=train_dataset._get_num_classes()).to(device)
    
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    
    # ================= KHỞI TẠO CHECKPOINT =================
    checkpoint_dir = "checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    start_epoch = 1
    print(f"Checkpoints will be saved to: ./{checkpoint_dir}/")

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"Resumed training from epoch {checkpoint['epoch']}")
        else:
            print(f"No checkpoint found at '{args.resume}', starting from scratch.")
    # ========================================================

    num_epochs = 32
    for epoch in range(start_epoch, num_epochs + 1):
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
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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

        # ================= LƯU CHECKPOINT Ở ĐÂY =================
        # 1. Lưu định kỳ mỗi 2 epochs
        if epoch % 2 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"Đã lưu checkpoint định kỳ tại: {checkpoint_path}")

        # # 2. Lưu lại bản có Accuracy trên tập Validation cao nhất (Rất quan trọng)
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_acc': val_acc
        #     }, best_checkpoint_path)
        #     print(f"Đã cập nhật checkpoint TỐT NHẤT: {best_checkpoint_path} (Val Acc: {val_acc:.2f}%)")
        # ========================================================

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Train VideoSiglip2")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    main(args)
