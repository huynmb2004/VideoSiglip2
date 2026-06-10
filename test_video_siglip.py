import torch
from transformers import Siglip2VisionConfig
from siglip2 import Siglip2VisionModel
from video_siglip import (
    HybridTemporalModule,
    Siglip2VideoVisionWrapper,
    apply_lora_to_vision_encoder,
    freeze_base_and_unfreeze_temporal_lora
)

def test_video_siglip():
    # 1. Initialize a small dummy config for testing
    config = Siglip2VisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_patches=196,
        patch_size=16,
        num_channels=3
    )
    
    print("1. Initializing base Siglip2VisionModel...")
    base_vision_model = Siglip2VisionModel(config)
    
    print("2. Applying LoRA to the base vision model...")
    lora_vision_model = apply_lora_to_vision_encoder(base_vision_model, r=8, lora_alpha=16)
    
    print("3. Initializing HybridTemporalModule...")
    # Parameters matches config.hidden_size
    temporal_module = HybridTemporalModule(
        embed_dim=config.hidden_size,
        local_kernel_size=3,
        num_summary_tokens=4, # Compress T frames down to 4 tokens
        num_transformer_layers=2,
        num_attention_heads=4
    )
    
    print("4. Wrapping into Siglip2VideoVisionWrapper...")
    video_model = Siglip2VideoVisionWrapper(
        vision_model=lora_vision_model,
        temporal_module=temporal_module
    )
    
    print("5. Applying gradient management strategy...")
    freeze_base_and_unfreeze_temporal_lora(video_model)
    
    # Check trainable parameters
    trainable_params = sum(p.numel() for p in video_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in video_model.parameters())
    print(f"   Trainable params: {trainable_params} || Total params: {total_params} || Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # 6. Forward pass with dummy data
    B, T, num_patches, patch_dim = 2, 8, 196, 768 # 14x14 patches of 16x16x3
    print(f"\n6. Running forward pass with dummy patchified data shape: (B={B}, T={T}, patches={num_patches}, patch_dim={patch_dim})...")
    dummy_video = torch.randn(B, T, num_patches, patch_dim)
    
    # Create dummy spatial shapes (since SigLIP2 uses them)
    # The spatial shape for each image is (H//patch_size, W//patch_size) -> (14, 14)
    dummy_spatial_shapes = torch.tensor([[14, 14], [14, 14]])
    
    # Run the model
    with torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", enabled=False):
        # Depending on if autocast is used, let's just do a normal pass.
        output = video_model(pixel_values=dummy_video, spatial_shapes=dummy_spatial_shapes)
        
    print(f"   Final Output Shape: {output.shape} (Expected: ({B}, {config.hidden_size}))")
    
    assert output.shape == (B, config.hidden_size), "Output shape mismatch!"
    print("\nSUCCESS! The Hybrid Temporal Module and LoRA integration are working correctly.")

if __name__ == "__main__":
    test_video_siglip()
