import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from pipeline.detector import Detector
from pipeline.scorer   import CLIPScorer
from pipeline.selector import TaskSelector


def get_images_with_object(class_name, ann_data, n=20):
    cat_name_to_id = {c["name"]: c["id"] for c in ann_data["categories"]}
    cat_id_to_imgs = defaultdict(set)
    for ann in ann_data["annotations"]:
        cat_id_to_imgs[ann["category_id"]].add(ann["image_id"])
    img_id_to_file = {img["id"]: img["file_name"] for img in ann_data["images"]}

    cat_id = cat_name_to_id.get(class_name)
    if cat_id is None:
        return []

    img_ids = list(cat_id_to_imgs[cat_id])
    random.shuffle(img_ids)
    paths = []
    for img_id in img_ids[:n*2]:
        fname = img_id_to_file.get(img_id, "")
        fpath = os.path.join(config.VAL_IMGS, fname)
        if os.path.exists(fpath):
            paths.append(fpath)
        if len(paths) >= n:
            break
    return paths


def evaluate(selector, n_images=20):
    import json
    with open(config.VAL_ANN) as f:
        ann_data = json.load(f)

    results_summary = {}
    total_acc = 0

    for task_name, task_info in config.TASK_DEFINITIONS.items():
        primary   = task_info["primary"]
        secondary = task_info["secondary"]

        relevant_imgs = get_images_with_object(primary[0], ann_data, n=n_images)

        if not relevant_imgs:
            results_summary[task_name] = 0
            continue

        correct = 0
        total   = 0

        for img_path in relevant_imgs:
            result = selector.select(img_path, task_name)
            total += 1
            if result["status"] == "success":
                sel = result["selected"]["class_name"]
                if sel in primary:         correct += 1.0
                elif sel in secondary:     correct += 0.5

        acc = correct / total if total > 0 else 0
        results_summary[task_name] = round(acc, 3)
        total_acc += acc

    avg = total_acc / len(results_summary)

    print("=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    for task, acc in results_summary.items():
        bar = "█" * int(acc * 25)
        print(f"  {task:<22} {acc:.0%}  {bar}")
    print("-" * 55)
    print(f"  {'AVERAGE':<22} {avg:.0%}")
    print("=" * 55)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(f"{config.OUTPUT_DIR}/results.json", "w") as f:
        json.dump({"tasks": results_summary, "average": avg}, f, indent=2)
    print(f"Results saved to {config.OUTPUT_DIR}/results.json")

    return results_summary, avg


if __name__ == "__main__":
    detector = Detector()
    scorer   = CLIPScorer()
    selector = TaskSelector(detector, scorer)
    evaluate(selector, n_images=20)