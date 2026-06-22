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

decord.bridge.set_bridge('torch')

from dataset import UCF101VideoDataset

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
