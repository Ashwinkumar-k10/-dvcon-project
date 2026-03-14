import os
import random
from pathlib import Path
import config
from pipeline.detector  import Detector
from pipeline.scorer    import CLIPScorer
from pipeline.selector  import TaskSelector

def main():
    print("Loading pipeline...")
    detector = Detector()
    scorer   = CLIPScorer()
    selector = TaskSelector(detector, scorer)

    all_imgs = list(Path(config.VAL_IMGS).glob("*.jpg"))
    print(f"Found {len(all_imgs)} images")

    for task_name in config.TASK_DEFINITIONS.keys():
        img_path = str(random.choice(all_imgs))
        result   = selector.select(img_path, task_name)

        print(f"Task: {task_name}")
        if result["status"] == "success":
            sel = result["selected"]
            print(f"  Selected : {sel['class_name']}")
            print(f"  Score    : {sel['final_score']:.2f}")
        else:
            print(f"  {result['status']}")

if __name__ == "__main__":
    main()