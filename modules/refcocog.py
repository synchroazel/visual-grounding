import json
import os
import pickle
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops.boxes import box_convert


class RefCOCOgSample:
    """
    An annotated image from RefCOCOg dataset.

    """

    def __init__(self,
                 img: Image.Image,
                 shape: tuple[int, int],
                 path: str,
                 img_id: str,
                 split: str,
                 category: str,
                 category_id: int,
                 sentences: list[str],
                 bbox: list[float],
                 segmentation: list[float]):
        self.img = img
        self.shape = shape
        self.path = path
        self.id = img_id
        self.split = split
        self.category = category
        self.category_id = category_id
        self.sentences = sentences
        self.bbox = bbox
        self.segmentation = segmentation

    def __repr__(self):
        return str(vars(self))


class RefCOCOg(Dataset):
    """
    Dataloader for RefCOCOg dataset.

    """

    def __init__(self, ds_path: str, split=None, transform=None):
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

        if split == 'train':
            self.refs = [ref for ref in self.refs if ref['split'] == 'train']
        elif split == 'val':
            self.refs = [ref for ref in self.refs if ref['split'] == 'val']
        elif split == 'test':
            self.refs = [ref for ref in self.refs if ref['split'] == 'test']

        self.size = len(self.refs)


    def __getitem__(self, idx: int):

        refs_data = self.refs[idx]

        for inst in self.instances['annotations']:
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

        bbox = torch.tensor(ann_data["bbox"])
        bbox = box_convert(bbox, "xywh", "xyxy").numpy()

        sample = {
            "img": pil_img,
            "shape": transforms.ToTensor()(pil_img).shape,
            "path": image_path,
            "img_id": refs_data["image_id"],
            "split": refs_data["split"],
            "category": self.categories[refs_data["category_id"]]["category"],
            "category_id": refs_data["category_id"],
            "sentences": [sentence["raw"].lower() for sentence in refs_data["sentences"]],
            "bbox": bbox,
            "segmentation": ann_data["segmentation"]
        }

        if self.transform:
            sample = self.transform(sample["img"], dtype=torch.float32)

        return sample

    def __len__(self):
        return self.size  # return the number of annotated images available using `refs(umd).p`
