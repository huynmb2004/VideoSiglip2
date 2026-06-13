import os
import torch
import decord
from torch.utils.data import Dataset
import numpy as np

decord.bridge.set_bridge('torch')

class KTHDebugDataset(Dataset):
    def __init__(self, root_dir, processor, split="train", num_samples=None, num_frames=8):
        """
        KTH Action Recognition Dataset (Debug/Sanity Check version)
        
        Args:
            root_dir (str): Path to the KTH dataset root.
            processor (AutoProcessor): Hugging Face processor.
            split (str): "train", "val", or "test".
            num_samples (int, optional): Truncate the dataset size for debugging.
            num_frames (int): Number of frames to extract (T=8).
        """
        self.root_dir = root_dir
        self.processor = processor
        self.split = split
        self.num_frames = num_frames
        
        # Exact KTH splitting logic based on person ID
        self.train_subjects = {"11", "12", "13", "14", "15", "16", "17", "18"}
        self.val_subjects = {"19", "20", "21", "23", "24", "25", "01", "04"}
        self.test_subjects = {"22", "02", "03", "05", "06", "07", "08", "09", "10"}
        
        if split == "train":
            self.target_subjects = self.train_subjects
        elif split == "val":
            self.target_subjects = self.val_subjects
        elif split == "test":
            self.target_subjects = self.test_subjects
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'.")
            
        self.video_paths = []
        self.action_classes = []
        
        # Traverse directory
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if not filename.endswith('.avi'):
                    continue
                    
                # Filename format: person01_boxing_d1_uncomp.avi
                parts = filename.split('_')
                if len(parts) >= 1 and parts[0].startswith("person"):
                    person_id = parts[0].replace("person", "")
                    
                    if person_id in self.target_subjects:
                        self.video_paths.append(os.path.join(class_dir, filename))
                        self.action_classes.append(class_name)
                        
        if num_samples is not None:
            self.video_paths = self.video_paths[:num_samples]
            self.action_classes = self.action_classes[:num_samples]
            
    def __len__(self):
        return len(self.video_paths)
        
    def _sample_frame_indices(self, total_frames):
        """Uniformly sample frames from the video."""
        if total_frames <= self.num_frames:
            # If video is too short, just repeat the last frame or pad
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        return indices
        
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        class_name = self.action_classes[idx]
        
        # Read frames using decord
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        
        indices = self._sample_frame_indices(total_frames)
        # decord returns [T, H, W, C]
        frames = vr.get_batch(indices) 
        
        # KTH is grayscale, so frames might be [T, H, W, 1] or [T, H, W, 3] where channels are identical.
        # Let's ensure it's a numpy array of shape [T, H, W, 3] for the processor
        frames_np = frames.numpy()
        if frames_np.ndim == 3:
            # [T, H, W] -> [T, H, W, 1]
            frames_np = np.expand_dims(frames_np, axis=-1)
        
        if frames_np.shape[-1] == 1:
            # Duplicate grayscale channel to RGB
            frames_np = np.repeat(frames_np, 3, axis=-1)
            
        # Ensure uint8
        if frames_np.dtype != np.uint8:
            frames_np = frames_np.astype(np.uint8)
            
        # Convert to list of numpy arrays for processor (shape: [H, W, 3])
        frames_list = [frame for frame in frames_np]
        
        prompt = f"A person is {class_name}"
        
        # Use AutoProcessor to handle resize, crop, normalization, and tokenization
        inputs = self.processor(
            text=[prompt],
            images=frames_list,
            padding="max_length",
            max_length=64, # Using max_length 64 as in SigLIP
            return_tensors="pt"
        )
        
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
            
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0) if "attention_mask" in inputs else None
        
        item = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }
        if attention_mask is not None:
            item["attention_mask"] = attention_mask
            
        return item

if __name__ == "__main__":
    from transformers import AutoProcessor
    model_name = "google/siglip2-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    
    root_dir = r"C:\Users\Admin\.cache\kagglehub\datasets\vafaeii\kth-action-recognition-dataset\versions\143"
    dataset = KTHDebugDataset(root_dir, processor, split="train", num_samples=4)
    print(f"Dataset Size: {len(dataset)}")
    
    if len(dataset) > 0:
        item = dataset[0]
        print(f"Pixel Values Shape: {item['pixel_values'].shape}")
        print(f"Input IDs Shape: {item['input_ids'].shape}")
