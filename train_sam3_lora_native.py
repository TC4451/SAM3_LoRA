
#!/usr/bin/env python3
"""
SAM3 LoRA Training Script

Validation Strategy (Following SAM3):
  - During training: Only compute validation LOSS (fast, no metrics)
  - After training: Run validate_sam3_lora.py for full metrics (mAP, cgF1) with NMS

This approach significantly speeds up training by avoiding expensive metric computation
during each epoch, while still monitoring overfitting via validation loss.

Multi-GPU Training:
  Single GPU:
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu

  Multi-GPU with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu
"""

import os
import argparse
import yaml
import json
import time
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, LoRALayer, apply_lora_to_model, save_lora_weights, count_parameters

from torchvision.transforms import v2
import pycocotools.mask as mask_utils  # Required for RLE mask decoding in COCO dataset
from sam3.train.masks_ops import rle_encode  # For encoding masks to RLE format

# Note: Evaluation modules (mAP, cgF1, NMS) are in validate_sam3_lora.py
# Training only computes validation loss, following SAM3's approach


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get the number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def format_duration(seconds: float) -> str:
    """Format duration in HH:MM:SS."""
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


_CUDA_TOTAL_MEM_GB_CACHE = {}


def get_cuda_memory_stats(device: torch.device) -> dict:
    """Get CUDA memory stats in GB for the specified device."""
    stats = {
        "gpu_mem_alloc_gb": 0.0,
        "gpu_mem_reserved_gb": 0.0,
        "gpu_mem_max_alloc_gb": 0.0,
        "gpu_mem_max_reserved_gb": 0.0,
        "gpu_mem_total_gb": 0.0,
    }
    if not torch.cuda.is_available():
        return stats
    if device.type != "cuda":
        return stats

    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    bytes_to_gb = 1024 ** 3
    if device_idx not in _CUDA_TOTAL_MEM_GB_CACHE:
        props = torch.cuda.get_device_properties(device_idx)
        _CUDA_TOTAL_MEM_GB_CACHE[device_idx] = props.total_memory / bytes_to_gb
    stats["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(device_idx) / bytes_to_gb
    stats["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device_idx) / bytes_to_gb
    stats["gpu_mem_max_alloc_gb"] = torch.cuda.max_memory_allocated(device_idx) / bytes_to_gb
    stats["gpu_mem_max_reserved_gb"] = torch.cuda.max_memory_reserved(device_idx) / bytes_to_gb
    stats["gpu_mem_total_gb"] = _CUDA_TOTAL_MEM_GB_CACHE[device_idx]
    return stats


class COCOSegmentDataset(Dataset):
    """Dataset class for COCO format segmentation data"""
    def __init__(self, data_dir, split="train"):
        """
        Args:
            data_dir: Root directory containing train/valid/test folders
            split: One of 'train', 'valid', 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        # self.split_dir = self.data_dir / split
        self.split_dir = self.data_dir

        # Load COCO annotations
        # ann_file = self.split_dir / "_annotations.coco.json"
        ann_file = self.data_dir / f"{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build index: image_id -> image info
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = sorted(list(self.images.keys()))

        # Build index: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Load categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        print(f"Loaded COCO dataset: {split} split")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Annotations: {len(self.coco_data['annotations'])}")
        print(f"  Categories: {self.categories}")

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.split_dir / img_info['file_name']
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get bbox - format is [x, y, width, height] in COCO format
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # Get class name from category_id
            category_id = ann.get("category_id", 0)
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)

            # Convert from COCO [x, y, w, h] to normalized [cx, cy, w, h] (CxCyWH)
            # SAM3 internally expects boxes in CxCyWH format normalized to [0, 1]
            x, y, w, h = bbox_coco
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Scale to resolution and normalize to [0, 1]
            box_tensor = torch.tensor([
                cx * scale_w / self.resolution,
                cy * scale_h / self.resolution,
                w * scale_w / self.resolution,
                h * scale_h / self.resolution,
            ], dtype=torch.float32)

            # Handle segmentation mask (polygon or RLE format)
            segment = None
            segmentation = ann.get("segmentation", None)

            if segmentation:
                try:
                    # Check if it's RLE format (dict) or polygon format (list)
                    if isinstance(segmentation, dict):
                        # RLE format: {"counts": "...", "size": [h, w]}
                        mask_np = mask_utils.decode(segmentation)
                    elif isinstance(segmentation, list):
                        # Polygon format: [[x1, y1, x2, y2, ...], ...]
                        # Convert polygon to RLE, then decode
                        rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                        rle = mask_utils.merge(rles)
                        mask_np = mask_utils.decode(rle)
                    else:
                        print(f"Warning: Unknown segmentation format: {type(segmentation)}")
                        segment = None
                        continue

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor

                except Exception as e:
                    print(f"Warning: Error processing mask for image {img_id}, ann {i}: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=i,
                segment=segment
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct Queries - one per unique category
        # Each query maps to only the objects of that category
        from collections import defaultdict

        # Group object IDs by their class name
        class_to_object_ids = defaultdict(list)
        for obj, class_name in zip(objects, object_class_names):
            class_to_object_ids[class_name.lower()].append(obj.object_id)

        # Create one query per category
        queries = []
        if len(class_to_object_ids) > 0:
            for query_text, obj_ids in class_to_object_ids.items():
                query = FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=0,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1
                    )
                )
                queries.append(query)
        else:
            # No annotations: create a single generic query
            query = FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.3):
    """
    Merge overlapping masks that likely represent the same object.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.3)

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute IoU
            intersection = (current_mask & binary_masks[j]).sum().item()
            union = (current_mask | binary_masks[j]).sum().item()
            iou = intersection / union if union > 0 else 0

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format for evaluation.

    OPTIMIZATION: Keep masks at native model output resolution (288×288)
    GT is downsampled to match, so no upsampling needed!

    Args:
        predictions_list: List of prediction dictionaries from the model
        image_ids: List of image IDs corresponding to predictions
        resolution: Mask resolution for evaluation (default: 288, model's native output)
        score_threshold: Minimum score threshold for predictions
        merge_overlaps: Whether to merge overlapping predictions (default: True)
        iou_threshold: IoU threshold for merging overlaps (default: 0.3)
        debug: Print debug information

    Returns:
        List of prediction dictionaries in COCO format
    """
    coco_predictions = []
    pred_id = 0

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Extract predictions
        logits = preds['pred_logits']  # [num_queries, 1]
        boxes = preds['pred_boxes']    # [num_queries, 4]
        masks = preds['pred_masks']    # [num_queries, H, W]

        scores = torch.sigmoid(logits).squeeze(-1)  # [num_queries]

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:  # Debug first image only
            print(f"  Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")

        # Convert masks to binary (apply sigmoid first, then threshold)
        binary_masks = (torch.sigmoid(masks) > 0.5).cpu()

        # Merge overlapping predictions to avoid over-segmentation penalty
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"  Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        # Encode masks to RLE (at native resolution - much faster!)
        if len(binary_masks) > 0:
            # Check if masks have content
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"  Mask shape: {binary_masks.shape}")
                print(f"  Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [cx, cy, w, h] to [x, y, w, h] in pixel coordinates
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                coco_predictions.append({
                    'image_id': int(img_id),
                    'category_id': 1,  # Single category for instance segmentation
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                })
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from SimpleSAM3Dataset.

    OPTIMIZATION: Downsample GT masks to match prediction resolution (288×288)
    instead of upsampling predictions to 1008×1008. Much faster!

    Args:
        dataset: SimpleSAM3Dataset instance
        image_ids: Optional list of specific image IDs to include
        mask_resolution: Resolution to downsample masks to (default: 288 to match model output)

    Returns:
        Dictionary in COCO format
    """
    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    # Scale factor for boxes (masks will be at mask_resolution, boxes scaled accordingly)
    scale_factor = mask_resolution / dataset.resolution

    for idx in indices:
        # Add image entry at mask resolution
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True  # Required for cgF1 evaluation
        })

        # Get datapoint
        datapoint = dataset[idx]

        # Add annotations
        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at mask_resolution
            cx, cy, bw, bh = (obj.bbox * mask_resolution).tolist()
            x, y, w, h = cx - bw / 2, cy - bh / 2, bw, bh

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            # Add segmentation if available - downsample to mask_resolution
            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    return coco_gt


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset, model_resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        # Process on GPU then immediately move to CPU to save memory
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Free GPU memory immediately after upsampling
        del masks_sigmoid, masks_upsampled
        torch.cuda.empty_cache()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at original size
            cx, cy, bw, bh = obj.bbox.tolist()
            w = bw * orig_w
            h = bh * orig_h
            x = cx * orig_w - w / 2
            y = cy * orig_h - h / 2

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


class SAM3TrainerNative:
    def __init__(self, config_path, multi_gpu=False, resume_checkpoint=None, auto_resume=False):
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.resume_checkpoint = Path(resume_checkpoint).expanduser() if resume_checkpoint else None
        self.auto_resume = auto_resume

        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Model
        print_rank0("Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,  # Tries to download from HF if checkpoint_path is None
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=False
        )

        # Apply LoRA
        print_rank0("Applying LoRA...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        stats = count_parameters(self.model)
        print_rank0(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        self.model.to(self.device)

        # Wrap model with DDP if multi-GPU
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # Frozen params (requires_grad=False) don't need this flag
            )
            print_rank0(f"Model wrapped with DistributedDataParallel")

        # Store reference to unwrapped model for accessing custom methods
        self._unwrapped_model = self.model.module if self.multi_gpu else self.model

        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Matcher & Loss
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        # Create loss functions with correct weights (from original SAM3 training config)
        # Note: These weights are for mask-based training
        loss_fns = [
            Boxes(weight_dict={
                "loss_bbox": 5.0,
                "loss_giou": 2.0
            }),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict={
                    "loss_ce": 20.0,
                    "presence_loss": 20.0
                },
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=200,
            ),
            Masks(
                weight_dict={
                    "loss_mask": 200.0,  # Much higher weight for mask loss!
                    "loss_dice": 10.0
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False
            )
        ]

        # Create one-to-many matcher for auxiliary outputs
        o2m_matcher = BinaryOneToManyMatcher(
            alpha=0.3,
            threshold=0.4,
            topk=4
        )

        # Use Sam3LossWrapper for proper loss computation
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",  # Use local normalization (no distributed training)
            normalize_by_valid_object_num=False,
        )

    def _extract_lora_state_dict(self, model: nn.Module):
        """Extract only LoRA weights from a model."""
        lora_state_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALayer):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.detach().cpu()
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.detach().cpu()
        return lora_state_dict

    def _move_optimizer_state_to_device(self):
        """Move optimizer state tensors to the current training device."""
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _resolve_resume_checkpoint(self, out_dir: Path, output_cfg: dict):
        """Resolve which checkpoint to use for resume, if any."""
        resume_from_cfg = (
            self.config.get("training", {}).get("resume_from")
            or output_cfg.get("resume_from")
        )
        explicit_resume_requested = self.resume_checkpoint is not None or bool(resume_from_cfg)
        resume_path = self.resume_checkpoint or (
            Path(resume_from_cfg).expanduser() if resume_from_cfg else None
        )

        if resume_path is not None:
            if not resume_path.is_absolute():
                resume_path = (self.config_path.parent / resume_path).resolve()
            if resume_path.exists():
                return resume_path
            if explicit_resume_requested:
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            return None

        auto_resume_enabled = bool(output_cfg.get("auto_resume", False)) or self.auto_resume
        if not auto_resume_enabled:
            return None

        resume_name = str(output_cfg.get("resume_checkpoint_name", "last_training_state.pt"))
        auto_path = out_dir / resume_name
        return auto_path if auto_path.exists() else None

    def _load_resume_checkpoint(self, checkpoint_path: Path):
        """
        Load resume checkpoint.
        Returns (start_epoch, best_val_loss, resumed_full_state).
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Full training-state checkpoint
        if (
            isinstance(checkpoint, dict)
            and "optimizer_state_dict" in checkpoint
            and "lora_state_dict" in checkpoint
        ):
            self._unwrapped_model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._move_optimizer_state_to_device()
            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            return start_epoch, best_val_loss, True

        # Fallback: LoRA-only weights file (no optimizer/epoch state)
        is_lora_only = (
            isinstance(checkpoint, dict)
            and len(checkpoint) > 0
            and all(k.endswith(".lora_A") or k.endswith(".lora_B") for k in checkpoint.keys())
        )
        if is_lora_only:
            self._unwrapped_model.load_state_dict(checkpoint, strict=False)
            return 0, float("inf"), False

        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint_path}. "
            "Expected a full training state checkpoint or LoRA-only weights."
        )

    def _save_resume_checkpoint(
        self,
        checkpoint_path: Path,
        model_to_save: nn.Module,
        epoch_num: int,
        best_val_loss: float,
    ):
        """Save full training state for resume (atomic write)."""
        checkpoint = {
            "epoch": int(epoch_num),
            "best_val_loss": float(best_val_loss),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lora_state_dict": self._extract_lora_state_dict(model_to_save),
            "config_path": str(self.config_path),
            "time": time.time(),
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)

    def train(self):
        # Get data directory from config (should point to directory containing train/valid folders)
        data_dir = self.config["training"]["data_dir"]

        # Load datasets using COCO format
        print_rank0(f"\nLoading training data from {data_dir}...")
        train_ds = COCOSegmentDataset(data_dir=data_dir, split="train")

        # Check if validation data exists
        has_validation = False
        val_ds = None

        try:
            print_rank0(f"\nLoading validation data from {data_dir}...")
            val_ds = COCOSegmentDataset(data_dir=data_dir, split="valid")
            if len(val_ds) > 0:
                has_validation = True
                print_rank0(f"Found validation data: {len(val_ds)} images")
            else:
                print_rank0(f"Validation dataset is empty.")
                val_ds = None
        except Exception as e:
            print_rank0(f"Could not load validation data: {e}")
            val_ds = None

        if not has_validation:
            val_ds = None

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None

        if self.multi_gpu:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.world_size,
                    rank=get_rank(),
                    shuffle=False
                )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=True
        )

        if has_validation:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=self.config["training"].get("num_workers", 0),
                pin_memory=True
            )
        else:
            val_loader = None

        self.model.train()

        # Weights from a standard SAM config roughly
        weight_dict = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0
        }

        epochs = self.config["training"]["num_epochs"]
        start_epoch = 0
        best_val_loss = float('inf')
        log_every_n_steps = int(self.config["training"].get("log_every_n_steps", 50))
        if log_every_n_steps <= 0:
            log_every_n_steps = 50
        print_metrics_to_console = bool(self.config["training"].get("print_metrics_to_console", False))

        if has_validation:
            print_rank0(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("⚠️  No validation data found - training without validation")

        if self.multi_gpu:
            print_rank0(f"Effective batch size: {self.config['training']['batch_size']} x {self.world_size} = {self.config['training']['batch_size'] * self.world_size}")

        # Helper to move BatchedDatapoint to device
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        # Create output directory
        output_cfg = self.config["output"]
        out_dir = Path(output_cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "training_metrics.jsonl"
        global_batch_size = self.config["training"]["batch_size"] * (self.world_size if self.multi_gpu else 1)
        save_resume_checkpoint = bool(output_cfg.get("save_resume_checkpoint", True))
        resume_checkpoint_name = str(output_cfg.get("resume_checkpoint_name", "last_training_state.pt"))
        resume_checkpoint_path = out_dir / resume_checkpoint_name

        # Resume support: restore LoRA weights + optimizer + epoch from full checkpoint.
        resume_path = self._resolve_resume_checkpoint(out_dir, output_cfg)
        resumed_full_state = False
        if resume_path is not None:
            loaded_start_epoch, loaded_best_val, resumed_full_state = self._load_resume_checkpoint(resume_path)
            start_epoch = loaded_start_epoch
            best_val_loss = loaded_best_val
            if resumed_full_state:
                print_rank0(
                    f"Resumed full training state from: {resume_path} "
                    f"(next epoch: {start_epoch + 1}, best_val_loss: {best_val_loss:.6f})"
                )
            else:
                print_rank0(
                    f"Loaded LoRA weights from resume path: {resume_path} "
                    "(optimizer/epoch state not found; training starts at epoch 1)."
                )

        print_rank0(f"Starting training for {epochs} epochs...")
        if start_epoch > 0:
            print_rank0(f"Continuing from epoch {start_epoch + 1}/{epochs}")
        if start_epoch >= epochs:
            print_rank0(
                f"Resume epoch ({start_epoch}) is already >= configured num_epochs ({epochs}). "
                "Nothing to train."
            )
            if self.multi_gpu:
                cleanup_distributed()
            return

        # Optional config snapshot for reproducibility (one per run)
        save_config_each_run = bool(output_cfg.get("save_config_each_run", True))
        config_snapshot_name_template = str(
            output_cfg.get("config_snapshot_name", "config_{run_id}.yaml")
        )
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        config_snapshot_path = None
        if is_main_process() and save_config_each_run:
            try:
                config_snapshot_name = config_snapshot_name_template.format(
                    run_id=run_id,
                    config_name=self.config_path.name,
                    config_stem=self.config_path.stem,
                )
            except Exception as e:
                print_rank0(
                    f"Warning: Invalid config snapshot template '{config_snapshot_name_template}': {e}. "
                    "Falling back to default template."
                )
                config_snapshot_name = f"config_{run_id}.yaml"

            config_snapshot_path = out_dir / config_snapshot_name
            config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            if self.config_path.exists():
                shutil.copy2(self.config_path, config_snapshot_path)
            else:
                with open(config_snapshot_path, "w") as f:
                    yaml.safe_dump(self.config, f, sort_keys=False)
            print_rank0(f"Saved run config snapshot: {config_snapshot_path}")

        # Optional per-epoch checkpointing (in addition to best/last checkpoints)
        save_every_epoch = bool(output_cfg.get("save_every_epoch", False))
        epoch_ckpt_subdir = output_cfg.get("epoch_ckpt_subdir", "")
        epoch_ckpt_name_template = str(
            output_cfg.get("epoch_ckpt_name", "epoch_{epoch:03d}_lora_weights.pt")
        )
        try:
            keep_last_n_epoch_ckpts = int(output_cfg.get("keep_last_n_epoch_ckpts", 0))
        except (TypeError, ValueError):
            keep_last_n_epoch_ckpts = 0
        keep_last_n_epoch_ckpts = max(0, keep_last_n_epoch_ckpts)

        epoch_ckpt_dir = out_dir / epoch_ckpt_subdir if epoch_ckpt_subdir else out_dir
        if save_every_epoch:
            epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)

        saved_epoch_ckpt_paths = []
        epoch_ckpt_template_error_logged = False

        try:
            epoch_ckpt_dir_display = str(epoch_ckpt_dir.relative_to(out_dir))
        except ValueError:
            epoch_ckpt_dir_display = str(epoch_ckpt_dir)
        if epoch_ckpt_dir_display in ("", "."):
            epoch_ckpt_target_display = epoch_ckpt_name_template
        else:
            epoch_ckpt_target_display = f"{epoch_ckpt_dir_display}/{epoch_ckpt_name_template}"

        def save_per_epoch_checkpoint(model_to_save, epoch_num: int):
            nonlocal epoch_ckpt_template_error_logged
            if not save_every_epoch:
                return

            try:
                ckpt_name = epoch_ckpt_name_template.format(epoch=epoch_num)
            except Exception as e:
                if not epoch_ckpt_template_error_logged:
                    print_rank0(
                        f"Warning: Invalid epoch checkpoint name template '{epoch_ckpt_name_template}': {e}. "
                        "Falling back to default template."
                    )
                    epoch_ckpt_template_error_logged = True
                ckpt_name = f"epoch_{epoch_num:03d}_lora_weights.pt"

            epoch_ckpt_path = epoch_ckpt_dir / ckpt_name
            epoch_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_lora_weights(model_to_save, str(epoch_ckpt_path))
            if len(saved_epoch_ckpt_paths) == 0 or saved_epoch_ckpt_paths[-1] != epoch_ckpt_path:
                saved_epoch_ckpt_paths.append(epoch_ckpt_path)

            if keep_last_n_epoch_ckpts > 0:
                while len(saved_epoch_ckpt_paths) > keep_last_n_epoch_ckpts:
                    old_path = saved_epoch_ckpt_paths.pop(0)
                    if old_path.exists():
                        old_path.unlink()

        def log_metrics(record: dict):
            if not is_main_process():
                return
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if is_main_process():
            run_meta = {
                "event": "run_start",
                "time": time.time(),
                "run_id": run_id,
                "epochs": epochs,
                "start_epoch": start_epoch,
                "resumed_full_state": resumed_full_state,
                "resume_checkpoint_path": str(resume_path) if resume_path is not None else None,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds) if has_validation and val_ds is not None else 0,
                "batch_size_per_gpu": self.config["training"]["batch_size"],
                "world_size": self.world_size,
                "global_batch_size": global_batch_size,
                "log_every_n_steps": log_every_n_steps,
                "config_path": str(self.config_path),
                "save_resume_checkpoint": save_resume_checkpoint,
                "resume_checkpoint_name": resume_checkpoint_name,
            }
            if config_snapshot_path is not None:
                run_meta["config_snapshot_path"] = str(config_snapshot_path)
            if torch.cuda.is_available() and self.device.type == "cuda":
                device_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_idx)
                run_meta["gpu_name"] = props.name
                run_meta["gpu_total_mem_gb"] = round(props.total_memory / (1024 ** 3), 2)
                if print_metrics_to_console:
                    print_rank0(
                        f"Logging every {log_every_n_steps} steps | GPU: {props.name} "
                        f"({run_meta['gpu_total_mem_gb']:.2f} GB)"
                    )
            elif print_metrics_to_console:
                print_rank0(f"Logging every {log_every_n_steps} steps")
            if print_metrics_to_console:
                print_rank0(f"Detailed metrics log: {log_path}")
            log_metrics(run_meta)

        run_start_time = time.time()
        for epoch in range(start_epoch, epochs):
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            # Track training losses for this epoch
            train_losses = []
            train_loss_sum = 0.0
            epoch_start_time = time.time()
            last_iter_end_time = epoch_start_time

            # Only show progress bar on rank 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            num_train_batches = len(train_loader)
            for step_idx, batch_dict in enumerate(pbar, start=1):
                iter_start_time = time.time()
                data_wait_time = iter_start_time - last_iter_end_time
                input_batch = batch_dict["input"]

                # Move to device
                input_batch = move_to_device(input_batch, self.device)

                # Forward pass
                # outputs_list is SAM3Output, we need to pass the whole thing to loss_wrapper
                outputs_list = self.model(input_batch)

                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget (one per stage)
                find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                # Move targets to device
                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                # Add matcher indices to outputs (required by Sam3LossWrapper)
                # Use SAM3Output.iteration_mode to properly iterate over outputs
                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        # stage_targets is a single target dict, replicate for all steps
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            # Compute indices for main output
                            outputs["indices"] = self.matcher(outputs, targets)

                            # Also add indices to auxiliary outputs if they exist
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                # Compute loss using Sam3LossWrapper
                # This handles num_boxes calculation and proper weighting
                loss_dict = self.loss_wrapper(outputs_list, find_targets)

                # Extract total loss
                total_loss = loss_dict[CORE_LOSS_KEY]

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Track training loss
                step_loss = total_loss.item()
                train_losses.append(step_loss)
                train_loss_sum += step_loss
                running_train_loss = train_loss_sum / step_idx
                iter_time = time.time() - iter_start_time
                epoch_elapsed = time.time() - epoch_start_time
                progress = step_idx / max(1, num_train_batches)
                eta_seconds = (epoch_elapsed / step_idx) * max(0, num_train_batches - step_idx)
                samples_per_second = (step_idx * global_batch_size) / max(epoch_elapsed, 1e-6)

                gpu_stats = get_cuda_memory_stats(self.device)
                pbar.set_postfix({
                    "loss": f"{step_loss:.3f}",
                    "avg": f"{running_train_loss:.3f}",
                    "it_s": f"{iter_time:.2f}",
                    "gpu_gb": f"{gpu_stats['gpu_mem_alloc_gb']:.1f}",
                })

                if is_main_process() and (
                    step_idx == 1 or
                    step_idx % log_every_n_steps == 0 or
                    step_idx == num_train_batches
                ):
                    if print_metrics_to_console:
                        print_rank0(
                            f"[Train] epoch {epoch+1}/{epochs} step {step_idx}/{num_train_batches} "
                            f"({progress * 100:.1f}%) loss={step_loss:.6f} avg_loss={running_train_loss:.6f} "
                            f"iter={iter_time:.2f}s data_wait={data_wait_time:.2f}s "
                            f"samples/s={samples_per_second:.2f} eta={format_duration(eta_seconds)} "
                            f"gpu_alloc={gpu_stats['gpu_mem_alloc_gb']:.2f}GB "
                            f"gpu_reserved={gpu_stats['gpu_mem_reserved_gb']:.2f}GB"
                        )
                    log_metrics({
                        "event": "train_step",
                        "time": time.time(),
                        "epoch": epoch + 1,
                        "step": step_idx,
                        "steps_per_epoch": num_train_batches,
                        "progress_pct": round(progress * 100, 2),
                        "loss": step_loss,
                        "running_loss": running_train_loss,
                        "iter_time_sec": iter_time,
                        "data_wait_sec": data_wait_time,
                        "epoch_elapsed_sec": epoch_elapsed,
                        "eta_sec": eta_seconds,
                        "samples_per_second": samples_per_second,
                        **gpu_stats,
                    })
                last_iter_end_time = time.time()

            # Calculate average training loss for this epoch
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

            # Validation (only compute loss - no metrics, like SAM3)
            if has_validation and val_loader is not None:
                self.model.eval()
                val_losses = []
                val_loss_sum = 0.0
                val_start_time = time.time()
                num_val_batches = len(val_loader)

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Validation", disable=not is_main_process())

                    for val_step_idx, batch_dict in enumerate(val_pbar, start=1):
                        val_iter_start_time = time.time()
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)

                        # Forward pass
                        outputs_list = self.model(input_batch)

                        # Prepare targets
                        find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                        # Move targets to device
                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        # Add matcher indices to outputs (required by Sam3LossWrapper)
                        with SAM3Output.iteration_mode(
                            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                                stage_targets_list = [stage_targets] * len(stage_outputs)
                                for outputs, targets in zip(stage_outputs, stage_targets_list):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(aux_out, targets)

                        # Compute loss using Sam3LossWrapper
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]

                        val_step_loss = total_loss.item()
                        val_losses.append(val_step_loss)
                        val_loss_sum += val_step_loss
                        running_val_loss = val_loss_sum / val_step_idx
                        val_iter_time = time.time() - val_iter_start_time
                        val_elapsed = time.time() - val_start_time
                        val_progress = val_step_idx / max(1, num_val_batches)
                        val_eta_seconds = (val_elapsed / val_step_idx) * max(0, num_val_batches - val_step_idx)
                        gpu_stats = get_cuda_memory_stats(self.device)
                        val_pbar.set_postfix({
                            "val_loss": f"{val_step_loss:.3f}",
                            "avg": f"{running_val_loss:.3f}",
                            "it_s": f"{val_iter_time:.2f}",
                            "gpu_gb": f"{gpu_stats['gpu_mem_alloc_gb']:.1f}",
                        })

                        if is_main_process() and (
                            val_step_idx == 1 or
                            val_step_idx % log_every_n_steps == 0 or
                            val_step_idx == num_val_batches
                        ):
                            if print_metrics_to_console:
                                print_rank0(
                                    f"[Valid] epoch {epoch+1}/{epochs} step {val_step_idx}/{num_val_batches} "
                                    f"({val_progress * 100:.1f}%) val_loss={val_step_loss:.6f} "
                                    f"avg_val_loss={running_val_loss:.6f} iter={val_iter_time:.2f}s "
                                    f"eta={format_duration(val_eta_seconds)} "
                                    f"gpu_alloc={gpu_stats['gpu_mem_alloc_gb']:.2f}GB "
                                    f"gpu_reserved={gpu_stats['gpu_mem_reserved_gb']:.2f}GB"
                                )
                            log_metrics({
                                "event": "val_step",
                                "time": time.time(),
                                "epoch": epoch + 1,
                                "step": val_step_idx,
                                "steps_per_epoch": num_val_batches,
                                "progress_pct": round(val_progress * 100, 2),
                                "val_loss": val_step_loss,
                                "running_val_loss": running_val_loss,
                                "iter_time_sec": val_iter_time,
                                "epoch_elapsed_sec": val_elapsed,
                                "eta_sec": val_eta_seconds,
                                **gpu_stats,
                            })

                avg_val_loss = sum(val_losses) / len(val_losses)

                # Synchronize val_loss across all processes for consistent best model selection
                if self.multi_gpu:
                    val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                print_rank0(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                if is_main_process():
                    epoch_time = time.time() - epoch_start_time
                    run_elapsed = time.time() - run_start_time
                    gpu_stats = get_cuda_memory_stats(self.device)
                    if print_metrics_to_console:
                        print_rank0(
                            f"[EpochSummary] epoch {epoch+1}/{epochs} train_loss={avg_train_loss:.6f} "
                            f"val_loss={avg_val_loss:.6f} epoch_time={format_duration(epoch_time)} "
                            f"run_elapsed={format_duration(run_elapsed)} "
                            f"gpu_max_alloc={gpu_stats['gpu_mem_max_alloc_gb']:.2f}GB "
                            f"gpu_max_reserved={gpu_stats['gpu_mem_max_reserved_gb']:.2f}GB"
                        )
                    log_metrics({
                        "event": "epoch_summary",
                        "time": time.time(),
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "epoch_time_sec": epoch_time,
                        "run_elapsed_sec": run_elapsed,
                        **gpu_stats,
                    })

                # Save models based on validation loss (only on rank 0)
                if is_main_process():
                    # Get underlying model from DDP wrapper
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))
                    save_per_epoch_checkpoint(model_to_save, epoch + 1)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_lora_weights(model_to_save, str(out_dir / "best_lora_weights.pt"))
                        print(f"✓ New best model saved (val_loss: {avg_val_loss:.6f})")

                    if save_resume_checkpoint:
                        self._save_resume_checkpoint(
                            checkpoint_path=resume_checkpoint_path,
                            model_to_save=model_to_save,
                            epoch_num=epoch + 1,
                            best_val_loss=best_val_loss,
                        )

                    # Log to file
                    with open(out_dir / "val_stats.json", "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss
                        }) + "\n")

                torch.cuda.empty_cache()

                # Back to training mode
                self.model.train()
            else:
                # No validation - just save model each epoch (only on rank 0)
                if is_main_process():
                    epoch_time = time.time() - epoch_start_time
                    run_elapsed = time.time() - run_start_time
                    gpu_stats = get_cuda_memory_stats(self.device)
                    if print_metrics_to_console:
                        print_rank0(
                            f"[EpochSummary] epoch {epoch+1}/{epochs} train_loss={avg_train_loss:.6f} "
                            f"epoch_time={format_duration(epoch_time)} run_elapsed={format_duration(run_elapsed)} "
                            f"gpu_max_alloc={gpu_stats['gpu_mem_max_alloc_gb']:.2f}GB "
                            f"gpu_max_reserved={gpu_stats['gpu_mem_max_reserved_gb']:.2f}GB"
                        )
                    log_metrics({
                        "event": "epoch_summary",
                        "time": time.time(),
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "epoch_time_sec": epoch_time,
                        "run_elapsed_sec": run_elapsed,
                        **gpu_stats,
                    })
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))
                    save_per_epoch_checkpoint(model_to_save, epoch + 1)
                    if save_resume_checkpoint:
                        self._save_resume_checkpoint(
                            checkpoint_path=resume_checkpoint_path,
                            model_to_save=model_to_save,
                            epoch_num=epoch + 1,
                            best_val_loss=best_val_loss,
                        )

        # Synchronize before final save
        if self.multi_gpu:
            dist.barrier()

        # Final save (only on rank 0)
        if is_main_process():
            if has_validation:
                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (best validation loss)")
                print(f"  - last_lora_weights.pt (last epoch)")
                if save_resume_checkpoint:
                    print(f"  - {resume_checkpoint_name} (full training resume state)")
                if save_every_epoch:
                    print(f"  - {epoch_ckpt_target_display} (one file per epoch)")
                print(f"\n📊 To compute full metrics (mAP, cgF1) with NMS:")
                print(f"   python validate_sam3_lora.py \\")
                print(f"     --config <config_path> \\")
                print(f"     --weights {out_dir}/best_lora_weights.pt \\")
                print(f"     --val_data_dir <data_dir>/valid")
                print(f"{'='*80}")
            else:
                # If no validation, copy last to best
                last_path = out_dir / "last_lora_weights.pt"
                best_path = out_dir / "best_lora_weights.pt"
                if last_path.exists():
                    shutil.copy(last_path, best_path)

                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (copy of last epoch)")
                print(f"  - last_lora_weights.pt (last epoch)")
                if save_resume_checkpoint:
                    print(f"  - {resume_checkpoint_name} (full training resume state)")
                if save_every_epoch:
                    print(f"  - {epoch_ckpt_target_display} (one file per epoch)")
                print(f"\nℹ️  No validation data - consider adding data/valid/ for better model selection")
                print(f"{'='*80}")

        # Cleanup distributed training
        if self.multi_gpu:
            cleanup_distributed()

def launch_distributed_training(args):
    """Launch training with multiple GPUs using torchrun subprocess."""
    import subprocess
    import sys

    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching distributed training on GPUs: {devices}")
    print(f"Number of processes: {num_gpus}")

    # Build the command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port", str(args.master_port),
        sys.argv[0],  # This script
        "--config", args.config,
        "--device", *map(str, devices),
        "--_launched_by_torchrun"  # Internal flag to indicate we're in subprocess
    ]
    if args.resume is not None:
        cmd.extend(["--resume", args.resume])
    if args.auto_resume:
        cmd.append("--auto-resume")

    # Set environment variable for visible devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str

    # Run the subprocess
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU (default GPU 0):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Single GPU (specific GPU):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 1

  Multi-GPU (GPUs 0 and 1):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1

  Multi-GPU (GPUs 0, 2, 3):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 2 3

  Multi-GPU (all 4 GPUs):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1 2 3
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_lora_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID(s) to use. Single value for single GPU, multiple values for multi-GPU. "
             "Example: --device 0 (single GPU), --device 0 1 2 (3 GPUs)"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to full training-state checkpoint to resume (restores LoRA + optimizer + epoch)."
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from output.resume_checkpoint_name in output_dir if it exists."
    )
    parser.add_argument(
        "--_launched_by_torchrun",
        action="store_true",
        help=argparse.SUPPRESS  # Hidden argument for internal use
    )
    args = parser.parse_args()

    # Determine if multi-GPU training is requested
    num_devices = len(args.device)
    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ

    if num_devices > 1 and not is_torchrun_subprocess:
        # Multi-GPU requested but not yet in torchrun - launch it
        launch_distributed_training(args)
    else:
        # Single GPU or already in torchrun subprocess
        multi_gpu = num_devices > 1 and is_torchrun_subprocess

        if not multi_gpu and num_devices == 1:
            # Single GPU mode - set the device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
            print(f"Using single GPU: {args.device[0]}")

        trainer = SAM3TrainerNative(
            args.config,
            multi_gpu=multi_gpu,
            resume_checkpoint=args.resume,
            auto_resume=args.auto_resume,
        )
        trainer.train()
