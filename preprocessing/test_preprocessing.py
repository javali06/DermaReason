import cv2
import matplotlib.pyplot as plt

from hair_removal import remove_hair
from normalize import preprocess_image

IMAGE_PATH = r"C:\Projects\DermaReason\dataset\raw\images\ISIC_0024306.jpg"

# Load image
img = cv2.imread(IMAGE_PATH)

# Hair removal
hair_removed = remove_hair(img)

# Normalization
processed = preprocess_image(hair_removed)

# Display results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(
    cv2.cvtColor(hair_removed, cv2.COLOR_BGR2RGB)
)
ax[1].set_title("Hair Removed")
ax[1].axis("off")

ax[2].imshow(processed)
ax[2].set_title("Normalized")
ax[2].axis("off")

plt.tight_layout()
plt.show()

print("Shape:", processed.shape)
print("Min Pixel Value:", processed.min())
print("Max Pixel Value:", processed.max())