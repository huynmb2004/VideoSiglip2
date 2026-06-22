import os
import torch
import random
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import decord

decord.bridge.set_bridge('torch')
logger = logging.getLogger(__name__)

class UCF101VideoDataset(Dataset):
    """
    Custom Dataset for UCF101 Video Action Recognition.
    Reads videos using decord and generates SigLIP text prompts.
    """
    def __init__(
        self,
        base_dir: str,
        annotation_dir: str,
        processor,
        split: int = 1,
        mode: str = 'train',
        num_frames: int = 8,
    ):
        """
        Args:
            base_dir (str): Base directory of the UCF101 dataset.
            annotation_dir (str): Directory containing classInd.txt, trainlist01.txt, etc.
            processor: Hugging Face AutoProcessor for SigLIP2.
            split (int): Split to use (1, 2, or 3).
            mode (str): 'train', 'val', or 'test'.
            num_frames (int): Number of frames to sample per video.
        """
        self.base_dir = base_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        self.split = split
        self.mode = mode
        self.num_frames = num_frames
        
        # Load label string to original ID map
        self.class_to_id = self._load_class_mapping()
            
        self.unique_labels = sorted(list(self.class_to_id.keys()))
        
        # Re-map IDs to contiguous 0..N-1 range
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
        self.video_list = self._load_split_list()

    def _load_class_mapping(self):
        classInd_path = os.path.join(self.annotation_dir, 'classInd.txt')
        class_to_id = {}
        with open(classInd_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    class_id = int(parts[0]) - 1
                    class_name = parts[1]
                    class_to_id[class_name] = class_id
        return class_to_id

    def _load_split_list(self):
        video_list = []
        if self.mode == 'train':
            list_file = os.path.join(self.annotation_dir, f'trainlist0{self.split}.txt')
            with open(list_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        vid_path = parts[0]
                        class_name = vid_path.split('/')[0]
                        if class_name in self.class_to_id:
                            label_id = self.label_to_id[class_name]
                            video_list.append((vid_path, label_id))
        else:
            # test/val sử dụng file testlist
            list_file = os.path.join(self.annotation_dir, f'testlist0{self.split}.txt')
            with open(list_file, 'r') as f:
                for line in f:
                    vid_path = line.strip()
                    if vid_path:
                        class_name = vid_path.split('/')[0]
                        if class_name in self.class_to_id:
                            label_id = self.label_to_id[class_name]
                            video_list.append((vid_path, label_id))
        return video_list

    def __len__(self):
        return len(self.video_list)

    def _get_num_classes(self):
        return len(self.unique_labels)

    def _get_frame_indices(self, total_frames: int):
        if total_frames <= self.num_frames:
            return np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        seg_size = total_frames / self.num_frames
        indices = []
        for i in range(self.num_frames):
            start = int(i * seg_size)
            end = int((i + 1) * seg_size)
            if self.mode == 'train':
                idx = random.randint(start, max(start, end - 1))
            else:
                idx = start + (end - start) // 2
            indices.append(idx)
        return np.array(indices)

    def __getitem__(self, idx):
        vid_path, label_id = self.video_list[idx]
        label_str = self.id_to_label[label_id]
        video_path = os.path.join(self.base_dir, vid_path)
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = self._get_frame_indices(total_frames)
            frames = vr.get_batch(frame_indices)
        except Exception as e:
            # logger.warning(f"Error reading video {video_path}: {e}")
            frames = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.uint8).permute(0, 2, 3, 1)

        text_prompt = f"A video of a person performing {label_str}"
        frames_np = [frame.numpy() for frame in frames]
        
        inputs = self.processor(
            images=frames_np, 
            text=text_prompt, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=64
        )
        
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
            
        input_ids = inputs["input_ids"].squeeze(0)
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
    
    BASE_DIR = "/media/lqngoc38/data/UCF-101/"
    ANNOTATION_DIR = "/media/lqngoc38/data/UCF-101/annotations/ucfTrainTestlist/"
    
    if os.path.exists(BASE_DIR) and os.path.exists(ANNOTATION_DIR):
        print("Loading Processor...")
        processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        
        print("Initializing Dataset...")
        dataset = UCF101VideoDataset(
            base_dir=BASE_DIR,
            annotation_dir=ANNOTATION_DIR,
            processor=processor,
            num_frames=8,
            mode='train'
        )
        
        print(f"Dataset Size: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample Output:")
            print(f"Pixel Values Shape: {sample['pixel_values'].shape}")
            print(f"Input IDs Shape: {sample['input_ids'].shape}")
            if sample.get('attention_mask') is not None:
                print(f"Attention Mask Shape: {sample['attention_mask'].shape}")
            print(f"Label ID: {sample['label_id']} (Maps to: {dataset.unique_labels[sample['label_id'].item()]})")
    else:
        print(f"Could not find dataset at {BASE_DIR} or {ANNOTATION_DIR}.")

