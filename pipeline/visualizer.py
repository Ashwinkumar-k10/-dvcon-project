import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import config

CLASS_COLORS = {
    "wine glass"   : (255, 100, 100),
    "cup"          : (100, 180, 255),
    "bottle"       : (255, 180, 50),
    "knife"        : (255, 80,  80),
    "spoon"        : (80,  200, 120),
    "scissors"     : (200, 100, 255),
    "baseball bat" : (255, 200, 0),
    "tennis racket": (0,   200, 200),
    "book"         : (150, 150, 255),
    "remote"       : (255, 150, 100),
    "fork"         : (100, 255, 150),
    "bowl"         : (200, 200, 100),
    "cell phone"   : (100, 200, 255),
    "default"      : (0,   255, 100),
}

def visualize(image_path, result, save_path=None):
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    try:
        font_big   = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_big = font_small = ImageFont.load_default()

    # Draw all detected objects faint grey
    for det in result.get("all_scored", []):
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(180, 180, 180, 80),
            width=1
        )

    # Draw selected object bold colored box
    if result.get("status") == "success":
        sel         = result["selected"]
        x1,y1,x2,y2= sel["bbox"]
        name        = sel["class_name"]
        r, g, b     = CLASS_COLORS.get(name, CLASS_COLORS["default"])

        draw.rectangle(
            [x1, y1, x2, y2],
            fill   =(r, g, b, 40),
            outline=(r, g, b, 255),
            width  =4
        )

        label = f"  {name}  {sel['final_score']:.0%}  "
        tb    = draw.textbbox((x1, y1), label, font=font_small)
        tw    = tb[2] - tb[0]
        th    = tb[3] - tb[1]
        ly    = max(0, y1 - th - 4)
        draw.rectangle(
            [x1, ly, x1+tw, ly+th+4],
            fill=(r, g, b, 220)
        )
        draw.text((x1, ly+2), label, fill=(0, 0, 0), font=font_small)

    # Task label at top
    draw.rectangle([0, 0, W, 40], fill=(20, 20, 20, 220))
    draw.text(
        (10, 8),
        f"  TASK: {result['task'].upper()}  "
        f"  Match: {result.get('match_type', '')}  ",
        fill=(255, 255, 0),
        font=font_big
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis("off")
    if result.get("status") == "success":
        sel = result["selected"]
        plt.title(
            f"Selected: {sel['class_name']}  |  "
            f"CLIP: {sel['clip_score']:.0%}  |  "
            f"Final: {sel['final_score']:.0%}",
            fontsize=13
        )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return img