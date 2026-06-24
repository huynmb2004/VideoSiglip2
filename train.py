import os
import argparse
import shutil
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import cv2  # <-- Added to write real temporary video blocks
import numpy as np
from torch.utils.data import DataLoader
import decord
from tqdm import tqdm
from transformers import AutoProcessor
from model import Siglip2LateFusionBaseline
from dataset import UCF101VideoDataset

decord.bridge.set_bridge('torch')

def create_mock_ucf101_environment(target_dir):
    """Generates structural folders, text manifests, and functional mock videos."""
    annotations_dir = os.path.join(target_dir, "annotations", "ucfTrainTestlist")
    os.makedirs(annotations_dir, exist_ok=True)
    
    mock_classes = ["ApplyEyeMakeUp", "Archery", "Basketball", "BikeRiding", "Bowling"]
    
    with open(os.path.join(annotations_dir, "classInd.txt"), "w") as f:
        for idx, cls in enumerate(mock_classes, 1):
            f.write(f"{idx} {cls}\n")
            
    train_lines = []
    test_lines = []
    
    for cls in mock_classes:
        class_video_dir = os.path.join(target_dir, cls)
        os.makedirs(class_video_dir, exist_ok=True)
        
        for i in range(1, 3):
            video_rel_path = f"{cls}/v_{cls}_g01_c0{i}.mp4"
            full_video_path = os.path.join(target_dir, video_rel_path)
            
            # Write a functional container file using OpenCV so decord doesn't complain about corrupt streams
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(full_video_path, fourcc, 10.0, (224, 224))
            for _ in range(16):  # Create a short 16-frame clip
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
                
            train_lines.append(f"{video_rel_path} {mock_classes.index(cls) + 1}\n")
            test_lines.append(f"{video_rel_path}\n")
            
    with open(os.path.join(annotations_dir, "trainlist01.txt"), "w") as f:
        f.writelines(train_lines)
    with open(os.path.join(annotations_dir, "testlist01.txt"), "w") as f:
        f.writelines(test_lines)

    return target_dir, annotations_dir


def main(args):
    base_dir = "/media/lqngoc38/data/UCF-101/"
    annotation_dir = "/media/lqngoc38/data/UCF-101/annotations/ucfTrainTestlist/"
    
    using_mock = False
    temp_sandbox = None

    if not (os.path.exists(base_dir) and os.path.exists(annotation_dir)):
        print("\n--- Real dataset paths not found. Building automated mock environment in temp folder... ---")
        temp_sandbox = tempfile.mkdtemp()
        base_dir, annotation_dir = create_mock_ucf101_environment(temp_sandbox)
        using_mock = True
        print(f"Temporary Workspace Setup at: {temp_sandbox}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model_name = "google/siglip2-base-patch16-224"
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name)
        
        print("Setting up Datasets and DataLoaders...")
        train_dataset = UCF101VideoDataset(base_dir=base_dir, annotation_dir=annotation_dir, processor=processor, num_frames=8, mode='train')
        val_dataset = UCF101VideoDataset(base_dir=base_dir, annotation_dir=annotation_dir, processor=processor, num_frames=8, mode='val')

        # Fixed: Set num_workers dynamically. Windows local prototyping needs 0 to prevent multi-worker deadlock bugs.
        num_workers = 0 if using_mock or os.name == 'nt' else 4
        print(f"Data pipeline configured with num_workers={num_workers}")

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        print(f"Train Dataset Size: {len(train_dataset)}")
        print(f"Validation Dataset Size: {len(val_dataset)}")
        
        print("Initializing Baseline Model...")
        model = Siglip2LateFusionBaseline(model_name=model_name, num_classes=train_dataset._get_num_classes()).to(device)
        
        criterion = nn.CrossEntropyLoss()
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=1e-4)
        
        checkpoint_dir = "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_acc = 0.0
        start_epoch = 1

        if args.resume:
            if os.path.isfile(args.resume):
                print(f"Loading checkpoint from {args.resume}...")
                checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_acc = checkpoint.get('val_acc', 0.0)
                print(f"Resumed training from epoch {checkpoint['epoch']}")

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

            if epoch % 2 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, best_checkpoint_path)

    finally:
        if using_mock and temp_sandbox and os.path.exists(temp_sandbox):
            print("\nCleaning up mock dataset temporary sandbox directories...")
            shutil.rmtree(temp_sandbox)
            print("Temporary sandbox storage cleared successfully.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Train VideoSiglip2")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    main(args)