import torch
import clip
from PIL import Image
import config

class CLIPScorer:
    def __init__(self):
        self.model, self.preprocess = clip.load(
            config.CLIP_MODEL,
            device=config.DEVICE
        )
        print(f"CLIP loaded: {config.CLIP_MODEL}")

    def score(self, image_path, detections, task_name):
        if not detections:
            return []

        task_info    = config.TASK_DEFINITIONS.get(task_name, {})
        descriptions = task_info.get("descriptions", [task_name])
        image        = Image.open(image_path).convert("RGB")
        W, H         = image.size

        # Encode all descriptions
        text_features = []
        for desc in descriptions:
            tokens = clip.tokenize([desc]).to(config.DEVICE)
            with torch.no_grad():
                tf = self.model.encode_text(tokens)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                text_features.append(tf)

        scored = []
        for det in detections:
            x1,y1,x2,y2 = [int(v) for v in det["bbox"]]
            crop = image.crop((x1,y1,x2,y2))
            if crop.width < 5 or crop.height < 5:
                continue

            ct = self.preprocess(crop).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                img_feat = self.model.encode_image(ct)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # Score against all descriptions
            scores    = [float((img_feat @ tf.T).squeeze())
                         for tf in text_features]
            best_clip = (max(scores) + 1) / 2

            # Size factor
            obj_area    = (x2-x1) * (y2-y1)
            img_area    = W * H
            size_factor = min(obj_area / img_area * 10, 1.0)

            # Position factor
            cx          = (x1+x2) / 2 / W
            cy          = (y1+y2) / 2 / H
            center_dist = ((cx-0.5)**2 + (cy-0.5)**2) ** 0.5
            pos_factor  = 1.0 - min(center_dist, 0.5)

            # Final combined score
            final = (
                det["confidence"] * 0.35 +
                best_clip         * 0.45 +
                size_factor       * 0.10 +
                pos_factor        * 0.10
            )

            scored.append({
                **det,
                "clip_score" : round(best_clip, 3),
                "size_factor": round(size_factor, 3),
                "pos_factor" : round(pos_factor, 3),
                "final_score": round(final, 3),
            })

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored