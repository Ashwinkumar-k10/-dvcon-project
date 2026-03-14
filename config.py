import os

# Auto detects if running on Kaggle or laptop
if os.path.exists("/kaggle"):
    COCO_ROOT  = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017"
    OUTPUT_DIR = "/kaggle/working/outputs"
    DEVICE     = "cuda"
else:
    COCO_ROOT  = "A:/coco"
    OUTPUT_DIR = "./outputs"
    DEVICE     = "cpu"   # laptop has no GPU

TRAIN_IMGS = f"{COCO_ROOT}/train2017"
VAL_IMGS   = f"{COCO_ROOT}/val2017"
TRAIN_ANN  = f"{COCO_ROOT}/annotations/instances_train2017.json"
VAL_ANN    = f"{COCO_ROOT}/annotations/instances_val2017.json"

MODEL_NAME  = "yolov8s.pt"
CLIP_MODEL  = "ViT-B/32"
CONF_THRESH = 0.15
IMG_SIZE    = 1280

TASK_CLASSES = [
    "wine glass", "cup", "bottle", "bowl",
    "knife", "fork", "spoon", "scissors",
    "baseball bat", "tennis racket",
    "remote", "cell phone", "book"
]

TASK_DEFINITIONS = {
    "serve a drink"  : {
        "descriptions": [
            "a wine glass or cup for drinking",
            "a glass or mug used to serve beverages",
            "a drinking vessel like a cup or wine glass",
        ],
        "primary"  : ["wine glass", "cup"],
        "secondary": ["bottle", "bowl"],
    },
    "pour liquid"    : {
        "descriptions": [
            "a bottle or container for pouring liquid",
            "a jug or bottle used to pour drinks",
            "a container for liquid pouring",
        ],
        "primary"  : ["bottle", "cup"],
        "secondary": ["bowl", "wine glass"],
    },
    "cut food"       : {
        "descriptions": [
            "a sharp knife for cutting food",
            "a kitchen knife or blade",
            "a cutting tool used in cooking",
        ],
        "primary"  : ["knife"],
        "secondary": ["scissors"],
    },
    "scoop food"     : {
        "descriptions": [
            "a spoon for scooping food",
            "a kitchen spoon or ladle",
            "a utensil for serving food",
        ],
        "primary"  : ["spoon"],
        "secondary": ["fork", "bowl"],
    },
    "spread on bread": {
        "descriptions": [
            "a knife for spreading butter on bread",
            "a spreading tool like a butter knife",
            "a flat knife used for spreading",
        ],
        "primary"  : ["knife"],
        "secondary": ["spoon"],
    },
    "pound or hammer": {
        "descriptions": [
            "a bat or heavy object for hammering",
            "a tool used to pound or strike",
            "a heavy object like a baseball bat",
        ],
        "primary"  : ["baseball bat"],
        "secondary": ["bottle"],
    },
    "clamp or grip"  : {
        "descriptions": [
            "scissors or pliers for gripping",
            "a tool used to clamp or hold objects",
            "a gripping tool like scissors",
        ],
        "primary"  : ["scissors"],
        "secondary": ["knife"],
    },
    "sweep floor"    : {
        "descriptions": [
            "a broom or racket for sweeping",
            "a long handled tool for cleaning floor",
            "an object used to sweep the floor",
        ],
        "primary"  : ["tennis racket"],
        "secondary": ["baseball bat"],
    },
    "write or draw"  : {
        "descriptions": [
            "a pen or pencil for writing",
            "a tool used to write or draw on paper",
            "a writing instrument like a pen",
        ],
        "primary"  : ["remote"],
        "secondary": ["cell phone"],
    },
    "support or prop": {
        "descriptions": [
            "a book or object used as a support",
            "something to prop or hold things up",
            "a flat object used for support",
        ],
        "primary"  : ["book"],
        "secondary": ["bottle"],
    },
    "open a bottle"  : {
        "descriptions": [
            "a knife or opener for opening bottles",
            "a tool used to open bottle caps",
            "an object for opening containers",
        ],
        "primary"  : ["knife"],
        "secondary": ["spoon", "scissors"],
    },
    "measure length" : {
        "descriptions": [
            "a ruler or book for measuring length",
            "a flat object used to measure distance",
            "a measuring tool or straight object",
        ],
        "primary"  : ["book"],
        "secondary": ["remote"],
    },
    "staple papers"  : {
        "descriptions": [
            "scissors or stapler for binding papers",
            "a tool used to fasten papers together",
            "an office tool for stapling",
        ],
        "primary"  : ["scissors"],
        "secondary": ["remote"],
    },
    "hang a picture" : {
        "descriptions": [
            "scissors or tool for hanging pictures",
            "an object used to hang or mount pictures",
            "a tool for putting pictures on walls",
        ],
        "primary"  : ["scissors"],
        "secondary": ["knife"],
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Config loaded. Device: {DEVICE}")
print(f"COCO Root: {COCO_ROOT}")