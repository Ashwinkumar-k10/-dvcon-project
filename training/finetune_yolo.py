from ultralytics import YOLO
import config

def finetune_yolo(yaml_path, output_dir):
    model = YOLO("yolov8s.pt")

    results = model.train(
        data    = yaml_path,
        epochs  = 50,
        imgsz   = 640,
        batch   = 32,
        lr0     = 0.001,
        patience= 15,
        device  = 0,
        project = output_dir,
        name    = "task_yolo",
        freeze  = 10,
        save    = True,
        plots   = True,
    )

    best = f"{output_dir}/task_yolo/weights/best.pt"
    print(f"Best model saved: {best}")
    return best

if __name__ == "__main__":
    finetune_yolo(
        yaml_path  = f"{config.OUTPUT_DIR}/task_dataset.yaml",
        output_dir = f"{config.OUTPUT_DIR}/finetuned"
    )