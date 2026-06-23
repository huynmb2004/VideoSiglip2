import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class HybridTemporalModule(nn.Module):
    """
    Hybrid Temporal Module for converting frame-level features into a video representation.
    """
    def __init__(
        self,
        embed_dim: int,
        local_kernel_size: int = 3,
        num_summary_tokens: int = 8,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_summary_tokens = num_summary_tokens

        # A. Local Temporal Branch: 1D Temporal Convolution
        # Expects input shape (B, embed_dim, T)
        # Using padding to keep temporal dimension size unchanged (if desired) before compression.
        padding = local_kernel_size // 2
        self.local_branch = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=local_kernel_size,
            padding=padding,
            groups=embed_dim  # depth-wise convolution as suggested to be lightweight
        )
        
        # B. Temporal Compression
        # Reduces T frames to M summary tokens
        self.compression = nn.AdaptiveAvgPool1d(num_summary_tokens)

        # C. Global Temporal Branch
        # Learn long-range dependencies across the M summary tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.global_branch = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # Learnable Class Token for Fusion (Alternative to Mean Pooling)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Frame feature sequence of shape (B, T, d)
        Returns:
            z_v: Final video representation of shape (B, d)
        """
        B, T, d = x.shape
        
        # Transpose for Conv1d and Pooling: (B, T, d) -> (B, d, T)
        x = x.transpose(1, 2)
        
        # A. Local Temporal Branch
        x_local = self.local_branch(x)
        # Residual connection over local branch (optional but good practice)
        x = x + x_local
        
        # B. Temporal Compression
        # (B, d, T) -> (B, d, M)
        x_compressed = self.compression(x)
        
        # Transpose back for Transformer: (B, d, M) -> (B, M, d)
        x_compressed = x_compressed.transpose(1, 2)
        
        # C. Global Temporal Branch
        # (B, M, d) -> (B, M, d)
        global_features = self.global_branch(x_compressed)
        
        # D. Fusion and Video Representation
        # Mean Pooling along the temporal axis
        # (B, M, d) -> (B, d)
        z_v = global_features.mean(dim=1)
        
        return z_v

class Siglip2VideoVisionWrapper(nn.Module):
    """
    Wrapper around Siglip2VisionModel to process video data.
    """
    def __init__(self, vision_model: nn.Module, temporal_module: HybridTemporalModule):
        super().__init__()
        self.vision_model = vision_model
        self.temporal_module = temporal_module

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: Video input of shape (B, T, C, H, W)
            spatial_shapes: Spatial shapes for the vision model, optional if not dynamically sized.
        Returns:
            z_v: Video representation of shape (B, d)
        """
        B, T = pixel_values.shape[:2]
        
        # Flatten batch and time dimensions to process frames independently
        pixel_values_flat = pixel_values.view(B * T, *pixel_values.shape[2:])
        
        if spatial_shapes is not None:
            # Replicate spatial shapes for each frame
            spatial_shapes_flat = spatial_shapes.repeat_interleave(T, dim=0)
        else:
            spatial_shapes_flat = None
            
        if pixel_attention_mask is not None:
            pixel_attention_mask_flat = pixel_attention_mask.view(B * T, -1)
        else:
            # Provide a dummy attention mask if None (all 1s)
            # Typically sequence length is (H//patch_size) * (W//patch_size)
            # We'll just infer it dynamically from spatial_shapes if possible, or provide a dummy.
            if spatial_shapes is not None:
                seq_len = spatial_shapes[0, 0] * spatial_shapes[0, 1]
            else:
                # Defaulting to 196 for 224x224 and patch_size 16
                seq_len = 196
            pixel_attention_mask_flat = torch.ones(B * T, seq_len, dtype=torch.long, device=pixel_values.device)
            
        # Extract spatial features using the base Vision Encoder
        # SigLIP2 Vision Model returns a tuple where the second element is often pooler_output
        # or the BaseModelOutputWithPooling object if return_dict=True.
        # We'll explicitly request return_dict=True to be safe.
        kwargs["return_dict"] = True
        vision_outputs = self.vision_model(
            pixel_values=pixel_values_flat,
            pixel_attention_mask=pixel_attention_mask_flat,
            spatial_shapes=spatial_shapes_flat,
            **kwargs
        )
        
        # The pooler_output (B*T, d) is the image representation
        frame_features = vision_outputs.pooler_output
        d = frame_features.shape[-1]
        
        # Reshape back to (B, T, d)
        frame_features = frame_features.view(B, T, d)
        
        # Pass through the Hybrid Temporal Module
        z_v = self.temporal_module(frame_features)
        
        return z_v

def apply_lora_to_vision_encoder(vision_model: nn.Module, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """
    Applies LoRA to the Attention projection layers of the Siglip2VisionModel using the peft library.
    """
    # The target modules typically include the Query, Key, Value, and Output projection layers 
    # in the self-attention blocks.
    # Looking at siglip2.py, the projection layers are named 'q_proj', 'k_proj', 'v_proj', 'out_proj'.
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=[], # We don't have other layers in the base model to save by default
    )
    
    lora_model = get_peft_model(vision_model, config)
    return lora_model

def freeze_base_and_unfreeze_temporal_lora(model: nn.Module):
    """
    Utility function to ensure:
    1. Base model parameters are frozen.
    2. LoRA parameters (A and B matrices) are unfrozen.
    3. HybridTemporalModule parameters are unfrozen.
    
    Note: If using `get_peft_model`, the base model is already frozen and LoRA layers are unfrozen.
    We just need to ensure the temporal module is unfrozen if it's part of a larger model structure.
    """
    # First, let peft handle the freezing of base model and unfreezing of LoRA
    # (This happens automatically in get_peft_model, but we can iterate to be explicit about the Wrapper)
    
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif "temporal_module" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
