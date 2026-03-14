from ultralytics import YOLO
import config

class Detector:
    def __init__(self, weights=config.MODEL_NAME):
        self.model = YOLO(weights)
        self.model.to(config.DEVICE)
        print(f"Detector loaded: {weights}")

    def detect(self, image_path):
        results = self.model(
            image_path,
            conf   = config.CONF_THRESH,
            imgsz  = config.IMG_SIZE,
            verbose= False
        )[0]

        detections = []
        if results.boxes is not None:
            for box, conf, cls_id in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy().astype(int)
            ):
                detections.append({
                    "class_name": self.model.names[cls_id],
                    "confidence": float(conf),
                    "bbox"      : box.tolist(),
                })
        return detections