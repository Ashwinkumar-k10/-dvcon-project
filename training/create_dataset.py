import os
import sys
import json
import shutil
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_task_dataset():
    OUTPUT_DIR = f"{config.OUTPUT_DIR}/task_dataset"
    os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/val",   exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/val",   exist_ok=True)

    def process_split(ann_file, img_dir, split):
        with open(ann_file) as f:
            coco = json.load(f)

        cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

        task_cat_ids = {
            cat_name_to_id[name]: i
            for i, name in enumerate(config.TASK_CLASSES)
            if name in cat_name_to_id
        }

        task_img_ids = set()
        for ann in coco["annotations"]:
            if ann["category_id"] in task_cat_ids:
                task_img_ids.add(ann["image_id"])

        img_info    = {img["id"]: img for img in coco["images"]}
        anns_per_img = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] in task_cat_ids:
                anns_per_img[ann["image_id"]].append(ann)

        processed = 0
        for img_id in task_img_ids:
            info  = img_info[img_id]
            fname = info["file_name"]
            W     = info["width"]
            H     = info["height"]
            src   = os.path.join(img_dir, fname)
            if not os.path.exists(src):
                continue

            shutil.copy2(src, f"{OUTPUT_DIR}/images/{split}/{fname}")

            label_path = f"{OUTPUT_DIR}/labels/{split}/{fname.replace('.jpg','.txt')}"
            with open(label_path, "w") as lf:
                for ann in anns_per_img[img_id]:
                    cls_id       = task_cat_ids[ann["category_id"]]
                    x, y, w, h   = ann["bbox"]
                    cx = (x + w/2) / W
                    cy = (y + h/2) / H
                    nw = w / W
                    nh = h / H
                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            processed += 1

        print(f"{split}: {processed} images processed")
        return processed

    print("Creating task dataset...")
    process_split(config.TRAIN_ANN, config.TRAIN_IMGS, "train")
    process_split(config.VAL_ANN,   config.VAL_IMGS,   "val")

    yaml_content = f"path: {OUTPUT_DIR}\ntrain: images/train\nval: images/val\nnc: {len(config.TASK_CLASSES)}\nnames:\n"
    for i, name in enumerate(config.TASK_CLASSES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = f"{OUTPUT_DIR}/task_dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset ready at: {OUTPUT_DIR}")
    print(f"YAML config: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    create_task_dataset()