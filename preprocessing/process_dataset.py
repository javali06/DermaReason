import os
import cv2
import numpy as np
from tqdm import tqdm

from hair_removal import remove_hair

INPUT_DIR = r"C:\Projects\DermaReason\dataset\raw\images"
OUTPUT_DIR = r"C:\Projects\DermaReason\dataset\processed\images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in tqdm(os.listdir(INPUT_DIR)):

    path = os.path.join(INPUT_DIR, filename)

    img = cv2.imread(path)

    if img is None:
        continue

    # Hair Removal
    img = remove_hair(img)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Save
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, filename),
        img
    )

print("Done!")