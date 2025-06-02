# import os
# import json
# import torch
# from PIL import Image
# from tqdm import tqdm
# from groundingdino.util.inference import load_model, predict, load_image
# from torchvision.transforms import Resize
# import torchvision.transforms as T

# # === Settings ===
# CONFIG_PATH = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
# IMAGE_DIR = "data/textvqa/images" 
# OUTPUT_DIR = "data/results"
# PATCH_DIR = os.path.join(OUTPUT_DIR, "patches")
# RESULT_PATH = os.path.join(OUTPUT_DIR, "results.json")
# PATCH_SCALES = [1.0, 1.5, 2.0]
# PATCH_SIZE = (224, 224)
# TOP_K = 100

# # === Initialize model ===
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = load_model(CONFIG_PATH, CHECKPOINT_PATH).to(device)
# resize_fn = Resize(PATCH_SIZE)

# # === Create output directory ===
# os.makedirs(PATCH_DIR, exist_ok=True)

# # === Load top 100 JSON entries ===
# with open("data/results/data_concepts.json", "r") as f:
#     all_data = json.load(f)
# entries = all_data[:TOP_K]

# # === Multi-scale patch extraction function ===
# def extract_scaled_patches(img_np, box, scales):
#     """
#     img_np: numpy image [H, W, C], RGB
#     box: (x1, y1, x2, y2)
#     scales: list of floats
#     returns: list of (bbox, PIL.Image, scale)
#     """
#     from PIL import Image

#     H, W, _ = img_np.shape
#     x1, y1, x2, y2 = box
#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2
#     bw = x2 - x1
#     bh = y2 - y1

#     patches = []
#     for s in scales:
#         nw, nh = int(bw * s), int(bh * s)
#         nx1 = max(cx - nw // 2, 0)
#         ny1 = max(cy - nh // 2, 0)
#         nx2 = min(cx + nw // 2, W)
#         ny2 = min(cy + nh // 2, H)

#         patch_np = img_np[ny1:ny2, nx1:nx2, :]  # crop numpy array
#         patch_pil = Image.fromarray(patch_np).resize(PATCH_SIZE)
#         patches.append(((nx1, ny1, nx2, ny2), patch_pil, s))
    
#     return patches

# # === Main loop ===
# results = []
# for entry in tqdm(entries):
#     image_id = entry["image_path"]
#     image_path = os.path.join(IMAGE_DIR, entry["image_path"])
#     concept_str = entry["concepts"].split("\n")[0].strip()

#     if not os.path.exists(image_path):
#         print(f"Missing image: {image_path}")
#         continue

#     img_src, img = load_image(image_path)
#     try:
#         boxes, _, phrases = predict(
#             model=model,
#             image=img,
#             caption=concept_str,
#             box_threshold=0.3,
#             text_threshold=0.25,
#             device=device
#         )
#     except Exception as e:
#         print(f"Failed on {image_id}: {e}")
#         continue

#     H, W, _ = img_src.shape
#     for i, (phrase, box) in enumerate(zip(phrases, boxes)):
#         x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [W, H, W, H])]
#         patches = extract_scaled_patches(img_src, (x1, y1, x2, y2), PATCH_SCALES)

#         for j, (coords, patch, scale) in enumerate(patches):
#             patch_filename = f"{image_id}_{phrase.replace(' ', '_')}_{scale:.1f}.jpg"
#             patch_path = os.path.join(PATCH_DIR, patch_filename)
#             patch.save(patch_path)

#             results.append({
#                 "image_id": image_id,
#                 "image_file": entry["image_path"],
#                 "concept": phrase,
#                 "scale": scale,
#                 "bbox": coords,
#                 "patch_file": patch_filename
#             })

# # === Save JSON results ===
# with open(RESULT_PATH, "w") as f:
#     json.dump(results, f, indent=2)

# print(f"Saved {len(results)} patches to: {PATCH_DIR}")

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from groundingdino.util.inference import load_model, predict, load_image
from torchvision.transforms import Resize
import torchvision.transforms as T
import numpy as np
import cv2

# === Settings ===
CONFIG_PATH = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
IMAGE_DIR = "data/textvqa/images"
OUTPUT_DIR = "data/results"
PATCH_DIR = os.path.join(OUTPUT_DIR, "patches_original")
RESULT_PATH = os.path.join(OUTPUT_DIR, "results_ori.json")
PATCH_SCALES = [1.0, 1.5, 2.0]
PATCH_SIZE = (224, 224)
TOP_K = 100

# === Initialize model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(CONFIG_PATH, CHECKPOINT_PATH).to(device)
resize_fn = Resize(PATCH_SIZE)

# === Create output directory ===
os.makedirs(PATCH_DIR, exist_ok=True)

# === Load top 100 JSON entries ===
with open("data/results/data_concepts.json", "r") as f:
    all_data = json.load(f)
entries = all_data[:TOP_K]

# === Multi-scale patch extraction function ===
def extract_scaled_patches(img_np, box, scales):
    H, W, _ = img_np.shape
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = x2 - x1
    bh = y2 - y1

    patches = []
    for s in scales:
        nw, nh = int(bw * s), int(bh * s)
        nx1 = max(cx - nw // 2, 0)
        ny1 = max(cy - nh // 2, 0)
        nx2 = min(cx + nw // 2, W)
        ny2 = min(cy + nh // 2, H)

        patch_np = img_np[ny1:ny2, nx1:nx2, :]
        patch_pil = Image.fromarray(patch_np).convert("RGB").resize(PATCH_SIZE)
        patches.append(((nx1, ny1, nx2, ny2), patch_pil, s))
    
    return patches

# === Main loop ===
results = []
for entry in tqdm(entries):
    image_id = entry["image_path"]
    image_path = os.path.join(IMAGE_DIR, entry["image_path"])
    concept_str = entry["concepts"].split("\n")[0].strip()

    if not os.path.exists(image_path):
        print(f"Missing image: {image_path}")
        continue

    img_src, img = load_image(image_path)
    try:
        boxes, _, phrases = predict(
            model=model,
            image=img,
            caption=concept_str,
            box_threshold=0.3,
            text_threshold=0.25,
            device=device
        )
    except Exception as e:
        print(f"Failed on {image_id}: {e}")
        continue

    H, W, _ = img_src.shape
    vis_img = img_src.copy()

    for i, (phrase, box) in enumerate(zip(phrases, boxes)):
        x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [W, H, W, H])]
        # draw bbox
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, phrase, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

        patches = extract_scaled_patches(img_src, (x1, y1, x2, y2), PATCH_SCALES)
        for j, (coords, patch, scale) in enumerate(patches):
            patch_filename = f"{image_id}_{phrase.replace(' ', '_')}_{scale:.1f}.jpg"
            patch_path = os.path.join(PATCH_DIR, patch_filename)
            patch.save(patch_path)

            results.append({
                "image_id": image_id,
                "image_file": entry["image_path"],
                "concept": phrase,
                "scale": scale,
                "bbox": coords,
                "patch_file": patch_filename
            })

    # save visualized image with boxes
    vis_save_path = os.path.join(PATCH_DIR, f"{image_id}_boxed.jpg")
    vis_img_pil = Image.fromarray(vis_img)
    vis_img_pil.save(vis_save_path)

# === Save JSON results ===
with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} patches and visualizations to: {PATCH_DIR}")