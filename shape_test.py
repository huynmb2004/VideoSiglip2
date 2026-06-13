import torch
from transformers import AutoProcessor, AutoModel
from dataset_debug import KTHDebugDataset
from video_siglip import Siglip2VideoVisionWrapper, HybridTemporalModule, apply_lora_to_vision_encoder
from torch.utils.data import DataLoader

def debug_forward_pass(base_model, vision_wrapper, dataloader):
    """Run exactly one forward pass without gradients and print exact tensor shapes."""
    print("--- Starting Debug Forward Pass ---")
    
    # Register Hooks to capture intermediate shapes
    hooks = []
    
    # 1. Hook for Spatial Encoder
    def spatial_hook(module, args, output):
        # output is BaseModelOutputWithPooling
        hidden_state = output.last_hidden_state
        # hidden_state: (B*T, num_patches, D)
        # We know B=batch_size, T=8 (from our dataloader)
        B = batch_size
        T = hidden_state.shape[0] // B
        num_patches = hidden_state.shape[1]
        D = hidden_state.shape[2]
        print(f"[DEBUG] After SigLIP Spatial Encoder: ({B}, {T}, {num_patches}, {D})")
    
    # vision_wrapper.vision_model is the PeftModel which wraps the original vision_model
    hooks.append(vision_wrapper.vision_model.register_forward_hook(spatial_hook))
    
    # 2. Hook for Local Temporal Branch
    def local_hook(module, args, output):
        # output of Conv1d is (B, D, T)
        print(f"[DEBUG] After Temporal Local Branch: {tuple(output.shape)}")
    
    hooks.append(vision_wrapper.temporal_module.local_branch.register_forward_hook(local_hook))
    
    # 3. Hook for Temporal Compression
    def compression_hook(module, args, output):
        # output of AdaptiveAvgPool1d is (B, D, M)
        print(f"[DEBUG] After Temporal Compression: {tuple(output.shape)}")
        
    hooks.append(vision_wrapper.temporal_module.compression.register_forward_hook(compression_hook))
    
    base_model.eval()
    vision_wrapper.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"]
            input_ids = batch["input_ids"]
            
            global batch_size
            batch_size = pixel_values.shape[0]
            
            print(f"[DEBUG] Input Video Shape: {tuple(pixel_values.shape)}")
            
            # Forward Vision
            z_v = vision_wrapper(pixel_values=pixel_values)
            print(f"[DEBUG] Final Video Representation z_v: {tuple(z_v.shape)}")
            
            # Forward Text
            text_outputs = base_model.text_model(input_ids=input_ids)
            z_t = text_outputs.pooler_output
            print(f"[DEBUG] Text Representation z_t: {tuple(z_t.shape)}")
            
            # Logits (assuming normalized dot product)
            z_v_norm = torch.nn.functional.normalize(z_v, p=2, dim=-1)
            z_t_norm = torch.nn.functional.normalize(z_t, p=2, dim=-1)
            
            # Scale with temp
            temp = torch.exp(base_model.logit_scale) if hasattr(base_model, "logit_scale") else 10.0
            logits = torch.matmul(z_v_norm, z_t_norm.t()) * temp
            print(f"[DEBUG] Logits Shape: {tuple(logits.shape)}")
            
            break # Only run exactly one batch
            
    # Remove hooks
    for h in hooks:
        h.remove()
    print("--- Debug Forward Pass Complete ---")

if __name__ == "__main__":
    model_name = "google/siglip2-base-patch16-224"
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    vision_model = base_model.vision_model
    vision_model = apply_lora_to_vision_encoder(vision_model, r=4) # lightweight r for debug
    
    temporal_module = HybridTemporalModule(embed_dim=vision_model.config.hidden_size)
    vision_wrapper = Siglip2VideoVisionWrapper(vision_model, temporal_module)
    
    root_dir = r"C:\Users\Admin\.cache\kagglehub\datasets\vafaeii\kth-action-recognition-dataset\versions\143"
    print("Initializing Dataset...")
    dataset = KTHDebugDataset(root_dir, processor, split="train", num_samples=4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    debug_forward_pass(base_model, vision_wrapper, dataloader)
