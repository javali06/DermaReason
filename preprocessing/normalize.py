import cv2
import numpy as np

def preprocess_image(image):
    """
    Resize + normalize image.
    """

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image.astype(np.float32)

    image = image / 255.0

    return image