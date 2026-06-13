import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import decord

decord.bridge.set_bridge('torch')

class UCF101VideoDataset(Dataset):
    """
    Custom Dataset for UCF101 Video Action Recognition.
    Reads videos using decord and generates SigLIP text prompts.
    """
    def __init__(
        self,
        csv_file: str,
        base_dir: str,
        processor,
        num_frames: int = 8,
        mode: str = 'train',
    ):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            base_dir (str): Base directory of the UCF101 dataset.
            processor: Hugging Face AutoProcessor for SigLIP2 (handles both image and text).
            num_frames (int): Number of frames to sample per video (T).
            mode (str): 'train', 'val', or 'test'. Determines the sampling strategy.
        """
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.processor = processor
        self.num_frames = num_frames
        self.mode = mode
        
        # Create a label to ID mapping
        self.unique_labels = sorted(self.data['label'].unique().tolist())
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        
    def __len__(self):
        return len(self.data)

    def _get_frame_indices(self, total_frames: int):
        """
        Divide the video into `num_frames` segments.
        If mode is 'train', randomly select 1 frame from each segment.
        If mode is 'val'/'test', select the center frame of each segment.
        """
        # Ensure we don't try to sample more frames than exist
        if total_frames < self.num_frames:
            # Pad by repeating the last frame if necessary, though ideally we just sample with replacement
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            return indices

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
        
        # Construct the absolute path
        # clip_path in CSV often starts with a slash, e.g. /train/Swing/...
        # We use os.path.normpath and os.path.join carefully.
        # Removing leading slash to ensure os.path.join works correctly
        if clip_path.startswith('/') or clip_path.startswith('\\'):
            clip_path = clip_path[1:]
        video_path = os.path.join(self.base_dir, clip_path)
        
        # 1. Read Video Frames
        try:
            # Initialize VideoReader
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Get indices
            frame_indices = self._get_frame_indices(total_frames)
            
            # Fetch frames (Returns shape: (T, H, W, C) in torch uint8)
            frames = vr.get_batch(frame_indices)
            
        except Exception as e:
            # Fallback for corrupted videos: return a dummy zero tensor
            print(f"Error reading video {video_path}: {e}")
            # The processor expects a list of numpy arrays or PIL Images or torch tensors (C, H, W)
            # Create dummy black frames (T, 3, 224, 224) roughly
            frames = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.uint8).permute(0, 2, 3, 1)

        # 2. Text Prompt Generation
        text_prompt = f"A video of a person performing {label_str}"
        
        # 3. Apply Processor
        # AutoProcessor for SigLIP handles images and texts
        # Images should be a list of 3D tensors (C, H, W) or numpy arrays (H, W, C)
        # We pass the frames as a list of numpy arrays (H, W, C)
        frames_np = [frame.numpy() for frame in frames]
        
        inputs = self.processor(
            images=frames_np, 
            text=text_prompt, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=64 # Typical max length for SigLIP text
        )
        
        # Extract pixel_values and input_ids
        # pixel_values shape from processor: (1, T, C, H, W) or (T, C, H, W)
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0) # (T, C, H, W)
            
        input_ids = inputs["input_ids"].squeeze(0) # (seq_len,)
        attention_mask = inputs["attention_mask"].squeeze(0) if "attention_mask" in inputs else None

        item = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "label_id": torch.tensor(label_id, dtype=torch.long)
        }
        if attention_mask is not None:
            item["attention_mask"] = attention_mask
            
        return item

if __name__ == "__main__":
    from transformers import AutoProcessor
    
    # Simple test to verify functionality
    base_dir = r"C:\Users\Admin\.cache\kagglehub\datasets\matthewjansen\ucf101-action-recognition\versions\4"
    train_csv = os.path.join(base_dir, "train.csv")
    
    if os.path.exists(train_csv):
        print("Loading Processor...")
        processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        
        print("Initializing Dataset...")
        dataset = UCF101VideoDataset(
            csv_file=train_csv,
            base_dir=base_dir,
            processor=processor,
            num_frames=8,
            mode='train'
        )
        
        print(f"Dataset Size: {len(dataset)}")
        sample = dataset[0]
        
        print("\nSample Output:")
        print(f"Pixel Values Shape: {sample['pixel_values'].shape}")
        print(f"Input IDs Shape: {sample['input_ids'].shape}")
        if sample['attention_mask'] is not None:
            print(f"Attention Mask Shape: {sample['attention_mask'].shape}")
        print(f"Label ID: {sample['label_id']} (Maps to: {dataset.unique_labels[sample['label_id'].item()]})")
    else:
        print(f"Could not find train.csv at {train_csv}. Please check the path.")
