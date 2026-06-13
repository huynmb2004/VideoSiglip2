import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoProcessor, AutoModel
from video_siglip import (
    HybridTemporalModule, 
    Siglip2VideoVisionWrapper, 
    apply_lora_to_vision_encoder, 
    freeze_base_and_unfreeze_temporal_lora
)
from loss import SiglipLoss
from dataset import UCF101VideoDataset

def evaluate(video_model, text_model, val_loader, unique_labels, processor, device, use_amp=True):
    video_model.eval()
    text_model.eval()
    
    print("Pre-computing text embeddings for Zero-shot Evaluation...")
    text_prompts = [f"A video of a person performing {label}" for label in unique_labels]
    inputs = processor(text=text_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            text_outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
            z_t_all = text_outputs.pooler_output # (num_classes, d)
            z_t_all = F.normalize(z_t_all, p=2, dim=-1)
            
    all_preds = []
    all_labels = []
    
    print("Extracting video features and computing similarities...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            label_ids = batch["label_id"].to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                z_v = video_model(pixel_values=pixel_values) # (B, d)
                z_v = F.normalize(z_v, p=2, dim=-1)
                
                # similarity matrix: (B, num_classes)
                logits = torch.matmul(z_v, z_t_all.t())
                
            top5_preds = logits.topk(min(5, logits.size(-1)), dim=-1).indices # (B, 5)
            
            all_preds.append(top5_preds.cpu())
            all_labels.append(label_ids.cpu())
            
    all_preds = torch.cat(all_preds, dim=0) # (N, 5)
    all_labels = torch.cat(all_labels, dim=0) # (N,)
    
    top1_correct = (all_preds[:, 0] == all_labels).sum().item()
    top5_correct = (all_preds == all_labels.unsqueeze(1)).any(dim=-1).sum().item()
    
    total = all_labels.size(0)
    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100
    print(f"--> Val Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"--> Val Top-5 Accuracy: {top5_acc:.2f}%")
    
    return top1_acc, top5_acc


def main():
    # --- Configurations ---
    base_dir = r"C:\Users\Admin\.cache\kagglehub\datasets\matthewjansen\ucf101-action-recognition\versions\4"
    train_csv = os.path.join(base_dir, "train.csv")
    val_csv = os.path.join(base_dir, "val.csv")
    
    model_name = "google/siglip2-base-patch16-224"
    batch_size = 4
    num_epochs = 5
    grad_accum_steps = 4
    num_frames = 8
    use_amp = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print("Loading Processor and Models from Hugging Face...")
    
    # Load Processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load Base SigLIP2 Model
    base_model = AutoModel.from_pretrained(model_name)
    
    vision_model = base_model.vision_model
    text_model = base_model.text_model
    
    # 1. Freeze Text Model completely
    for param in text_model.parameters():
        param.requires_grad = False
        
    # 2. Apply LoRA to Vision Encoder
    print("Applying LoRA to Vision Encoder...")
    vision_model = apply_lora_to_vision_encoder(vision_model, r=16, lora_alpha=32)
    
    # 3. Initialize Hybrid Temporal Module
    print("Initializing Hybrid Temporal Module...")
    temporal_module = HybridTemporalModule(
        embed_dim=vision_model.config.hidden_size,
        local_kernel_size=3,
        num_summary_tokens=num_frames // 2 if num_frames > 1 else 1, # Compress frames
        num_transformer_layers=2,
        num_attention_heads=8
    )
    
    # 4. Wrap the vision model
    video_model = Siglip2VideoVisionWrapper(vision_model, temporal_module)
    freeze_base_and_unfreeze_temporal_lora(video_model)
    
    # 5. Define Loss
    loss_fn = SiglipLoss(init_t=10.0, init_b=-10.0)
    
    # Move models to device
    video_model.to(device)
    text_model.to(device)
    loss_fn.to(device)
    
    # --- Dataset & DataLoader ---
    print("Setting up Datasets and DataLoaders...")
    train_dataset = UCF101VideoDataset(train_csv, base_dir, processor, num_frames=num_frames, mode='train')
    val_dataset = UCF101VideoDataset(val_csv, base_dir, processor, num_frames=num_frames, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # --- Optimizer & Scheduler ---
    print("Configuring Optimizer (Layer-wise LR)...")
    param_groups = [
        {"params": [p for n, p in video_model.named_parameters() if "temporal_module" in n and p.requires_grad], "lr": 1e-3},
        {"params": [p for n, p in video_model.named_parameters() if "lora_" in n and p.requires_grad], "lr": 1e-4},
        {"params": loss_fn.parameters(), "lr": 1e-4}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    
    total_steps = len(train_loader) * num_epochs // grad_accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # --- Training Loop ---
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        video_model.train()
        loss_fn.train()
        epoch_loss = 0.0
        
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                # 1. Video representations
                z_v = video_model(pixel_values=pixel_values) # (B, d)
                
                # 2. Text representations
                text_outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
                z_t = text_outputs.pooler_output # (B, d)
                
                # 3. Compute Sigmoid Loss
                loss = loss_fn(z_v, z_t)
                loss = loss / grad_accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            current_loss = loss.item() * grad_accum_steps
            epoch_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Completed. Avg Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation ---
        evaluate(video_model, text_model, val_loader, val_dataset.unique_labels, processor, device, use_amp)

if __name__ == "__main__":
    main()
