import cv2


path = r"C:\Projects\DermaReason\dataset\raw\images\ISIC_0024306.jpg"

img = cv2.imread(path)
img = cv2.resize(img, (224,224))
img = img / 255.0

print(img.shape)