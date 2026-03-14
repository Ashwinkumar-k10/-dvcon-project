import os
import sys
import json
import random
import torch
import torch.nn as nn
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TaskCLIPDataset(Dataset):
    def __init__(self, ann_file, img_dir, preprocess):
        self.samples    = []
        self.preprocess = preprocess

        with open(ann_file) as f:
            coco = json.load(f)

        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        img_info       = {img["id"]: img for img in coco["images"]}

        # Map object → task descriptions
        obj_to_descs = {}
        for task_name, info in config.TASK_DEFINITIONS.items():
            for obj in info["primary"] + info["secondary"]:
                if obj not in obj_to_descs:
                    obj_to_descs[obj] = []
                obj_to_descs[obj].extend(info["descriptions"])

        for ann in coco["annotations"]:
            cat_name = cat_id_to_name.get(ann["category_id"])
            if cat_name not in obj_to_descs:
                continue

            img_id = ann["image_id"]
            info   = img_info[img_id]
            fname  = os.path.join(img_dir, info["file_name"])
            if not os.path.exists(fname):
                continue

            x, y, w, h = ann["bbox"]
            desc = random.choice(obj_to_descs[cat_name])

            self.samples.append({
                "image_path": fname,
                "bbox"      : [x, y, x+w, y+h],
                "text"      : desc,
            })

        random.shuffle(self.samples)
        self.samples = self.samples[:5000]
        print(f"CLIP dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        img  = Image.open(s["image_path"]).convert("RGB")
        x1,y1,x2,y2 = [int(v) for v in s["bbox"]]
        crop = img.crop((x1, y1, x2, y2))
        if crop.width < 5 or crop.height < 5:
            crop = img
        image_tensor = self.preprocess(crop)
        text_tensor  = clip.tokenize([s["text"]], truncate=True)[0]
        return image_tensor, text_tensor


def finetune_clip(epochs=5, lr=1e-6):
    device = config.DEVICE
    print(f"Fine-tuning CLIP on {device}...")

    clip_model, preprocess = clip.load(config.CLIP_MODEL, device=device)

    dataset    = TaskCLIPDataset(config.TRAIN_ANN, config.TRAIN_IMGS, preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer  = torch.optim.Adam(clip_model.parameters(), lr=lr)
    clip_model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, texts) in enumerate(dataloader):
            images = images.to(device)
            texts  = texts.to(device)

            image_features = clip_model.encode_image(images)
            text_features  = clip_model.encode_text(texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

            logit_scale      = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text  = logits_per_image.t()

            labels = torch.arange(len(images)).to(device)
            loss   = (
                nn.CrossEntropyLoss()(logits_per_image, labels) +
                nn.CrossEntropyLoss()(logits_per_text,  labels)
            ) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} "
                      f"Batch {batch_idx}/{len(dataloader)} "
                      f"Loss={loss.item():.4f}")

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(dataloader):.4f}")

    clip_model.eval()

    save_path = f"{config.OUTPUT_DIR}/finetuned_clip.pt"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.save(clip_model.state_dict(), save_path)
    print(f"Fine-tuned CLIP saved: {save_path}")
    return save_path


if __name__ == "__main__":
    finetune_clip(epochs=5, lr=1e-6)