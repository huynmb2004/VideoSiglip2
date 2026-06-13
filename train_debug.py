import torch
import torch.optim as optim
from transformers import AutoProcessor, AutoModel
from torch.utils.data import DataLoader
from dataset_debug import KTHDebugDataset
from video_siglip import Siglip2VideoVisionWrapper, HybridTemporalModule, apply_lora_to_vision_encoder, freeze_base_and_unfreeze_temporal_lora
from loss import SiglipLoss

def test_overfitting_on_batch(base_model, vision_wrapper, dataloader, num_epochs=100):
    print(f"--- Starting Overfit-on-a-Batch Test for {num_epochs} Epochs ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_model = base_model.to(device)
    vision_wrapper = vision_wrapper.to(device)
    
    # Initialize SigLIP Loss
    criterion = SiglipLoss().to(device)
    
    # Freeze base model and unfreeze temporal module and LoRA
    for param in base_model.text_model.parameters():
        param.requires_grad = False
    freeze_base_and_unfreeze_temporal_lora(vision_wrapper)
    
    # Setup Optimizer
    trainable_params = list(vision_wrapper.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(trainable_params, lr=1e-3) # Higher LR for quick overfitting
    
    # Get exactly one batch
    batch = next(iter(dataloader))
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    
    base_model.eval() # Base model text encoder in eval mode
    vision_wrapper.train()
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        
        # 1. Forward Vision
        z_v = vision_wrapper(pixel_values=pixel_values)
        
        # 2. Forward Text (No gradients needed for text encoder since it's frozen)
        with torch.no_grad():
            text_outputs = base_model.text_model(input_ids=input_ids)
            z_t = text_outputs.pooler_output
            
        # 3. Compute Loss
        # SiglipLoss internally generates the labels (identity matrix)
        loss = criterion(z_v, z_t)
        
        # 5. Backward and Optimize
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}", flush=True)
            
    if loss.item() > 0.1:
        print("[WARNING] The loss did not converge near zero. There might be gradient flow issues or architecture bottlenecks!", flush=True)
    else:
        print("[SUCCESS] The model successfully overfit on the batch!", flush=True)

if __name__ == "__main__":
    model_name = "google/siglip2-base-patch16-224"
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    vision_model = base_model.vision_model
    # Apply LoRA
    vision_model = apply_lora_to_vision_encoder(vision_model, r=16)
    
    # Wrap with Temporal Module
    temporal_module = HybridTemporalModule(embed_dim=vision_model.config.hidden_size)
    vision_wrapper = Siglip2VideoVisionWrapper(vision_model, temporal_module)
    
    root_dir = r"C:\Users\Admin\.cache\kagglehub\datasets\vafaeii\kth-action-recognition-dataset\versions\143"
    print("Initializing Dataset...")
    # Get just 4 samples for the batch
    dataset = KTHDebugDataset(root_dir, processor, split="train", num_samples=4)
    
    # Collate fn will default to PyTorch's default_collate, but we must handle None gracefully if needed
    # (We fixed the None in our previous dataset implementation by simply not returning it)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    test_overfitting_on_batch(base_model, vision_wrapper, dataloader, num_epochs=15)
