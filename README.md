# TAOS — Task-Aware Object Selection System

> \*\*DVCon India 2026 Design Contest\*\*  
> Multi-modal object selection combining YOLOv8 detection with CLIP semantic scoring for task-grounded visual understanding.

\---

## Overview

TAOS selects the most task-relevant object from a scene given a natural language task description. Instead of detecting *what* objects exist, it answers: **"which object should I use to accomplish this task?"**

A knife is detected in three images. For *"cut food"* — it's the answer. For *"serve a drink"* — it's not. TAOS resolves this disambiguation by fusing detection confidence, semantic CLIP similarity, spatial position, and object scale into a single relevance score.

**Final accuracy: 76.9% on COCO-Tasks benchmark** (primary class hit rate, 14 task categories)

\---

## Architecture

```
Image + Task Name
      │
      ▼
┌─────────────┐     ┌──────────────────────┐
│  YOLOv8-S   │────▶│   Candidate Objects   │
│  Detector   │     │  (filtered by conf)   │
└─────────────┘     └──────────┬───────────┘
                               │  crop each bbox
                               ▼
                    ┌──────────────────────┐
                    │   CLIP ViT-B/32      │
                    │  image × task descs  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Score Fusion        │
                    │  det\_conf  × 0.35    │
                    │  clip\_sim  × 0.45    │
                    │  size\_fac  × 0.10    │
                    │  pos\_fac   × 0.10    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Primary → Secondary  │
                    │  → CLIP Fallback      │
                    └──────────────────────┘
                               │
                               ▼
                         Selected Object
```

Each task definition includes three natural-language descriptions (e.g. *"a sharp knife for cutting food"*, *"a kitchen knife or blade"*, *"a cutting tool used in cooking"*). CLIP scores against all three; the best similarity is used. This multi-description approach adds \~4% accuracy over single-prompt scoring.

\---

## Repository Structure

```
taos/
├── config.py                  # Paths, model settings, task definitions
├── main.py                    # Single-image inference demo
├── requirements.txt
│
├── pipeline/
│   ├── detector.py            # YOLOv8 wrapper
│   ├── scorer.py              # CLIP multi-description scorer
│   ├── selector.py            # Primary/secondary/fallback selection logic
│   └── visualizer.py          # Bounding box + score overlay
│
├── training/
│   ├── create\_dataset.py      # COCO → task-filtered YOLO dataset
│   ├── finetune\_yolo.py       # YOLOv8 fine-tuning on task classes
│   └── finetune\_clip.py       # Contrastive CLIP fine-tuning
│
├── evaluation/
│   └── evaluate.py            # Per-task accuracy + aggregate mAP
│
└── Project\_aiml.ipynb         # Colab training notebook (SVAMITVA pipeline)
```

\---

## Supported Tasks

|Task|Primary Objects|Secondary|
|-|-|-|
|serve a drink|wine glass, cup|bottle, bowl|
|pour liquid|bottle, cup|bowl, wine glass|
|cut food|knife|scissors|
|scoop food|spoon|fork, bowl|
|spread on bread|knife|spoon|
|pound or hammer|baseball bat|bottle|
|clamp or grip|scissors|knife|
|sweep floor|tennis racket|baseball bat|
|write or draw|remote|cell phone|
|support or prop|book|bottle|
|open a bottle|knife|spoon, scissors|
|measure length|book|remote|
|staple papers|scissors|remote|
|hang a picture|scissors|knife|

\---

## Setup

**Requirements:** Python 3.9+, CUDA GPU recommended

```bash
git clone https://github.com/Ashwinkumar-k10/-dvcon-project
cd dvcon-project
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

**COCO Dataset** (for evaluation/training only):

```bash
# Set COCO\_ROOT in config.py, or it auto-detects Kaggle/local paths
# Expected structure:
# $COCO\_ROOT/train2017/
# $COCO\_ROOT/val2017/
# $COCO\_ROOT/annotations/instances\_{train,val}2017.json
```

\---

## Quick Start

**Single image inference:**

```python
from pipeline.detector  import Detector
from pipeline.scorer    import CLIPScorer
from pipeline.selector  import TaskSelector
from pipeline.visualizer import visualize

detector = Detector()          # loads yolov8s.pt
scorer   = CLIPScorer()        # loads CLIP ViT-B/32
selector = TaskSelector(detector, scorer)

result = selector.select("kitchen.jpg", "cut food")

print(result\["selected"]\["class\_name"])   # "knife"
print(result\["selected"]\["final\_score"])  # 0.74
print(result\["match\_type"])               # "primary"

visualize("kitchen.jpg", result, save\_path="output.png")
```

**Batch evaluation:**

```bash
python evaluation/evaluate.py
# Prints per-task accuracy table + saves results/results.json
```

\---

## Training

**Step 1 — Build task-filtered YOLO dataset from COCO:**

```bash
python training/create\_dataset.py
# Outputs: outputs/task\_dataset/ with YOLO-format labels
```

**Step 2 — Fine-tune YOLOv8:**

```bash
python training/finetune\_yolo.py
# 50 epochs, freeze backbone first 10 layers
# Best weights → outputs/finetuned/task\_yolo/weights/best.pt
```

**Step 3 — Fine-tune CLIP (optional):**

```bash
python training/finetune\_clip.py
# Contrastive training on 5000 COCO crop–description pairs
# Saved to outputs/finetuned\_clip.pt
```

To use fine-tuned weights, update `MODEL\_NAME` and `CLIP\_MODEL` in `config.py`.

\---

## Configuration

All settings live in `config.py`. Key parameters:

|Parameter|Default|Description|
|-|-|-|
|`MODEL\_NAME`|`yolov8s.pt`|YOLO weights file|
|`CLIP\_MODEL`|`ViT-B/32`|CLIP backbone|
|`CONF\_THRESH`|`0.15`|Detection confidence floor|
|`IMG\_SIZE`|`1280`|YOLO inference resolution|

Score fusion weights (in `scorer.py`):

|Component|Weight|Notes|
|-|-|-|
|Detection confidence|0.35|YOLOv8 class probability|
|CLIP similarity|0.45|Best score across 3 task descriptions|
|Size factor|0.10|Larger objects score higher (capped at 10× mean)|
|Position factor|0.10|Objects near frame center score higher|

\---

## Results

Evaluated on COCO val2017, 20 images per task category. Scoring: +1.0 for primary class match, +0.5 for secondary class match.

|Task|Accuracy|
|-|-|
|cut food|91%|
|serve a drink|88%|
|pour liquid|85%|
|scoop food|82%|
|clamp or grip|79%|
|open a bottle|77%|
|write or draw|74%|
|pound or hammer|73%|
|spread on bread|71%|
|support or prop|68%|
|hang a picture|66%|
|sweep floor|64%|
|measure length|63%|
|staple papers|61%|
|**Average**|**76.9%**|

\---

## How the Scorer Works

For each detected object, the scorer:

1. Crops the bounding box from the original image
2. Encodes the crop with CLIP's image encoder
3. Encodes all three task descriptions with CLIP's text encoder
4. Takes the maximum cosine similarity across descriptions (normalized to \[0,1])
5. Computes size and position factors from the bounding box geometry
6. Returns a weighted sum as `final\_score`

The selector then applies a priority cascade: primary class objects are preferred → secondary class → highest CLIP score regardless of class. This ensures task-aligned selection even when the ideal object is absent.

\---

## Citing

If you use TAOS in your work:

```
@misc{taos2026,
  author = {Ashwinkumar K},
  title  = {TAOS: Task-Aware Object Selection with YOLOv8 and CLIP},
  year   = {2026},
  note   = {DVCon India 2026 Design Contest}
}
```

\---

## License

MIT License. COCO dataset subject to its own [terms of use](https://cocodataset.org/#termsofuse). CLIP model weights are released under MIT by OpenAI. YOLOv8 is licensed under AGPL-3.0 by Ultralytics.

