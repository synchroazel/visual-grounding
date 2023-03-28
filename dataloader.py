import os
import pickle
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RefCOCOgSample:
    """
    An annotated image from RefCOCOg dataset.

    """

    # TODO: add or remove attributes as needed.

    def __init__(self, img, shape, path, img_id, category_id, sentences, split):
        self.img = img
        self.shape = shape
        self.path = path
        self.id = img_id
        self.category_id = category_id
        self.sentences = sentences
        self.split = split

    def __repr__(self):
        return f"RefCOCOgSample(\n\tid={self.id}," \
               f"\n\tpath={self.path}," \
               f"\n\tshape={self.shape}," \
               f"\n\tcategory_id={self.category_id}," \
               f"\n\tsentences={self.sentences}," \
               f"\n\tsplit={self.split}" \
               f"\n)"


class RefCOCOg(Dataset):
    """
    Dataloader for RefCOCOg dataset.

    """

    def __init__(self, ds_path, transform=None):
        super(RefCOCOg, self).__init__()

        self.transform = transform

        self.ds_path = ds_path

        with open(f'{ds_path}/annotations/refs(umd).p', 'rb') as f:
            self.annotations = pickle.load(f)

        self.size = len(self.annotations)

    def __getitem__(self, idx):
        ann_data = self.annotations[idx]

        # print(ann_data)

        image_path = os.path.join(
            self.ds_path,
            "images",
            re.sub(r"_[0-9]+\.jpg", ".jpg", ann_data["file_name"])
            # "_".join(ann_data["file_name"].split("_")[:-1])+".jpg"
        )

        pil_img = Image.open(image_path)

        tensor_img = transforms.ToTensor()(pil_img)

        sample = RefCOCOgSample(
            img=tensor_img,
            shape=tensor_img.shape,
            path=image_path,
            img_id=ann_data["image_id"],
            category_id=ann_data["category_id"],
            sentences=[sentence["raw"].lower() for sentence in ann_data["sentences"]],
            split=ann_data["split"],
        )

        if self.transform:
            sample = self.transform(sample.img, dtype=torch.float32)

        return sample

    def __len__(self):
        return self.size  # return the number of annotated images available using `refs(umd).p`

        # return len(os.listdir(f"{self.ds_path}/images"))  # return the number of total images available

        # TODO: clarify the split to do - why only ~40k images are annotated?
