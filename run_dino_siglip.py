import os
import cv2
import torch
import numpy as np
from PIL import Image

# -----------------------------
# Grounding DINO imports
# -----------------------------
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict, annotate

# -----------------------------
# SigLIP imports
# -----------------------------
from transformers import AutoProcessor, AutoModel

# -----------------------------
# PATHS
# -----------------------------
TARGET_IMAGE = "data/targets/target.png"
QUERY_DIR = "data/queries"

DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "GroundingDINO/groundingdino_swint_ogc.pth"

# -----------------------------
# LOAD GROUNDING DINO
# -----------------------------
print("Loading Grounding DINO...")
cfg = SLConfig.fromfile(DINO_CONFIG)
dino = build_model(cfg)

ckpt = torch.load(DINO_WEIGHTS, map_location="cpu")
dino.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
dino.eval()

# -----------------------------
# LOAD SIGLIP
# -----------------------------
print("Loading SigLIP...")
SIGLIP_MODEL = "google/siglip-base-patch16-224"
siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL)
siglip_model.eval()

def embed_image(pil_img):
    inputs = siglip_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        emb = siglip_model.get_image_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)

# -----------------------------
# LOAD TARGET
# -----------------------------
image = cv2.imread(TARGET_IMAGE)
assert image is not None, "Failed to load target image"
h, w, _ = image.shape

# -----------------------------
# LOOP OVER QUERY SYMBOLS
# -----------------------------
results = []

for fname in sorted(os.listdir(QUERY_DIR)):
    if not fname.lower().endswith(".png"):
        continue

    symbol = os.path.splitext(fname)[0]
    query_path = os.path.join(QUERY_DIR, fname)

    print(f"\nProcessing symbol: {symbol}")

    query_img = Image.open(query_path).convert("RGB")
    query_emb = embed_image(query_img)

    # ---- Grounding DINO detect
    boxes, logits, phrases = predict(
        model=dino,
        image=image,
        caption=f"{symbol} safety symbol icon",
        box_threshold=0.3,
        text_threshold=0.25,
    )

    matches = 0

    for box in boxes:
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        crop = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_emb = embed_image(crop_pil)

        sim = float((query_emb @ crop_emb.T).item())
        if sim >= 0.85:
            matches += 1

    results.append((symbol, matches))
    print(f"Count: {matches}")

# -----------------------------
# OUTPUT
# -----------------------------
print("\nFINAL RESULTS")
for sym, cnt in results:
    print(f"{sym}: {cnt}")
