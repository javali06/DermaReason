import os
import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset


class HAMDataset(Dataset):

    def __init__(self, csv_file, image_dir):

        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_id = self.df.iloc[idx]["image_id"]
        label = self.df.iloc[idx]["label"]

        image_path = os.path.join(
            self.image_dir,
            image_id + ".jpg"
        )

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype("float32") / 255.0

        image = torch.tensor(
            image
        ).permute(2,0,1)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label