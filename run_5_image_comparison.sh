#!/bin/bash
# Visualize LoRA predictions on 5 images from real_wire.
# Keeps GPU impact low to avoid interfering with active training.

set -euo pipefail

DATA_DIR="../data/real_wire"
CONFIG_PATH="configs/cable_seg_config_3.yaml"
WEIGHTS_PATH="outputs/cable_lora_6/last_lora_weights.pt"
OUTPUT_DIR="comparison_outputs/real_wire_lora6_last"
NUM_IMAGES=10
MIN_FREE_MB=4500

if [ ! -f "$DATA_DIR/_annotations.coco.json" ]; then
  echo "Error: annotation file not found: $DATA_DIR/_annotations.coco.json"
  exit 1
fi

if [ ! -f "$WEIGHTS_PATH" ]; then
  echo "Error: weights file not found: $WEIGHTS_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "==============================================================="
echo "LoRA Prediction Visualization (real_wire)"
echo "==============================================================="
echo "Data dir: $DATA_DIR"
echo "Config:   $CONFIG_PATH"
echo "Weights:  $WEIGHTS_PATH"
echo "Mode:     single-model inference, low priority"
echo "GPU gate: run only if free VRAM >= ${MIN_FREE_MB}MB"
echo "Output:   $OUTPUT_DIR"
echo ""

mapfile -t IMAGES < <(
  python3 - "$DATA_DIR" "$NUM_IMAGES" <<'PY'
import json
import random
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
num_images = int(sys.argv[2])
ann_path = data_dir / "_annotations.coco.json"

with open(ann_path, "r") as f:
    coco = json.load(f)

existing = []
for img in coco.get("images", []):
    rel = img.get("file_name", "")
    if not rel:
        continue
    path = data_dir / rel
    if path.exists():
        existing.append(rel)

if not existing:
    raise SystemExit("No image files from annotation were found on disk.")

random.seed(4)
random.shuffle(existing)
for rel in existing[:num_images]:
    print(rel)
PY
)

if [ "${#IMAGES[@]}" -eq 0 ]; then
  echo "Error: no images selected from annotation."
  exit 1
fi

echo "Selected ${#IMAGES[@]} images:"
for img in "${IMAGES[@]}"; do
  echo "  - $img"
done
echo ""

for img in "${IMAGES[@]}"; do
  img_path="$DATA_DIR/$img"
  safe_name="$(echo "$img" | tr '/ ' '__')"
  output_path="$OUTPUT_DIR/pred_${safe_name}.png"

  echo "Processing: $img"
  if command -v nvidia-smi >/dev/null 2>&1; then
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [ -z "${free_mb:-}" ]; then
      free_mb=0
    fi
    if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
      echo "Skipping $img (free VRAM ${free_mb}MB < ${MIN_FREE_MB}MB to protect training)."
      echo ""
      continue
    fi
    echo "Free VRAM: ${free_mb}MB"
  fi

  # LoRA-only inference uses one model and is much lighter than base-vs-lora comparison.
  if ! nice -n 15 python3 inference_lora.py \
      --config "$CONFIG_PATH" \
      --weights "$WEIGHTS_PATH" \
      --image "$img_path" \
      --prompt cable \
      --output "$output_path" \
      --threshold 0.5 \
      --nms-iou 0.5 2>&1 | grep -E "Saved|Output saved|Detected objects|Max confidence|Error|Warning"; then
    echo "Warning: failed on $img (likely temporary GPU contention/OOM)."
  fi

  echo ""
done

echo "==============================================================="
echo "Comparison complete!"
echo "==============================================================="
echo "Check the outputs in: $OUTPUT_DIR/"
echo ""
ls -lh "$OUTPUT_DIR/"
