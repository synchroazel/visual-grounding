import json
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

    def __init__(self, img, shape, path, img_id, category, sentences, bbox, segmentation):
        self.img = img
        self.shape = shape
        self.path = path
        self.id = img_id
        self.category = category
        self.sentences = sentences
        self.bbox = bbox
        self.segmentation = segmentation

    def __repr__(self):
        return str(vars(self))


class RefCOCOg(Dataset):
    """
    Dataloader for RefCOCOg dataset.

    """

    def __init__(self, ds_path, transform=None):
        super(RefCOCOg, self).__init__()

        self.transform = transform

        self.ds_path = ds_path

        with open(f'{ds_path}/annotations/refs(umd).p', 'rb') as f:
            self.refs = pickle.load(f)

        with open(f"{ds_path}/annotations/instances.json", "r") as f:
            self.instances = json.load(f)

        self.categories = {
            item["id"]: {
                "supercategory": item["supercategory"],
                "category": item["name"]
            }
            for item in self.instances['categories']
        }

        self.size = len(self.refs)

    def __getitem__(self, idx):

        refs_data = self.refs[idx]

        for inst in self.instances['annotations']:
            # if inst['image_id'] == refs_data['image_id']:
            if inst['id'] == refs_data['ann_id']:
                ann_data = inst
                break

        image_path = os.path.join(
            self.ds_path,
            "images",
            re.sub(r"_[0-9]+\.jpg", ".jpg", refs_data["file_name"])
            # "_".join(ann_data["file_name"].split("_")[:-1])+".jpg"
        )

        pil_img = Image.open(image_path)

        tensor_img = transforms.ToTensor()(pil_img)

        sample = RefCOCOgSample(
            img=tensor_img,
            shape=tensor_img.shape,
            path=image_path,
            img_id=refs_data["image_id"],
            category=self.categories[refs_data["category_id"]]["category"],
            sentences=[sentence["raw"].lower() for sentence in refs_data["sentences"]],
            bbox=ann_data["bbox"],
            segmentation=ann_data["segmentation"]
        )

        if self.transform:
            sample = self.transform(sample.img, dtype=torch.float32)

        return sample

    def __len__(self):
        return self.size  # return the number of annotated images available using `refs(umd).p`

        # return len(os.listdir(f"{self.ds_path}/images"))  # return the number of total images available

        # TODO: clarify the split to do - why only ~40k images are annotated?
