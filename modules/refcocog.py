import json
import os
import pickle
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops.boxes import box_convert


class RefCOCOgSample:
    """
    An annotated image from RefCOCOg dataset.

    """

    def __init__(self,
                 img: Image.Image,
                 path: str,
                 img_id: str,
                 split: str,
                 category: str,
                 category_id: int,
                 sentences: list[str],
                 bbox: list[float],
                 segmentation: list[float]):
        self.img = img
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
    Dataset object for RefCOCOg dataset.

    """

    def __init__(self, ds_path: str, split=None, transform_img=None, transform_txt=None):
        super(RefCOCOg, self).__init__()

        self.transform_img = transform_img
        self.transform_txt = transform_txt

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

        self.instances = {inst['id']: inst for inst in self.instances['annotations']}

        if split == 'train':
            self.refs = [ref for ref in self.refs if ref['split'] == 'train']
        elif split == 'val':
            self.refs = [ref for ref in self.refs if ref['split'] == 'val']
        elif split == 'test':
            self.refs = [ref for ref in self.refs if ref['split'] == 'test']

        self.size = len(self.refs)

    def __getitem__(self, idx: int):

        # Get referenced data
        refs_data = self.refs[idx]

        # Get annotation data
        ann_data = self.instances[refs_data['ann_id']]

        # Get the image file path
        image_path = os.path.join(
            self.ds_path,
            "images",
            re.sub(r"_[0-9]+\.jpg", ".jpg", refs_data["file_name"])
        )

        # Load the image as a PIL image
        pil_img = Image.open(image_path)

        # Get the sentences from refs data
        sentences = [sentence["raw"].lower() for sentence in refs_data["sentences"]]

        # Apply transforms (if any) to the image and sentences
        if self.transform_img:
            pil_img = self.transform_img(pil_img)

        if self.transform_txt:
            sentences = [self.transform_txt(sentence) for sentence in sentences]

        # Get the bounding box
        bbox = torch.tensor(ann_data["bbox"])
        bbox = box_convert(bbox, "xywh", "xyxy").numpy()

        sample = {
            "img": pil_img,
            "path": image_path,
            "img_id": refs_data["image_id"],
            "split": refs_data["split"],
            "category": self.categories[refs_data["category_id"]]["category"],
            "category_id": refs_data["category_id"],
            "sentences": sentences,
            "bbox": bbox,
            "segmentation": ann_data["segmentation"]
        }

        return sample

    def __len__(self):
        return self.size  # return the number of annotated images available using `refs(umd).p`
