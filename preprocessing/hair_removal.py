import cv2
import numpy as np

def remove_hair(image):
    """
    Removes hair artifacts using BlackHat filtering + inpainting.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    hair_removed = cv2.inpaint(
        image,
        mask,
        1,
        cv2.INPAINT_TELEA
    )

    return hair_removed