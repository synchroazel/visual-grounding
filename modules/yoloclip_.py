import copy
import time

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

from modules.utilities import cosine_similarity, display_preds


class YoloClip:

    def __init__(self,
                 categories,
                 yolo_ver="yolov8x",
                 clip_ver="RN50",
                 device="cpu",
                 quiet=False):

        self.categories = copy.deepcopy(categories)
        self.yolo_ver = yolo_ver
        self.clip_ver = clip_ver
        self.yolo_model = YOLO(self.yolo_ver + ".pt")
        self.clip_model, self.clip_prep = clip.load(clip_ver, device="cpu")
        self.device = device
        self.quiet = quiet

        # model is loaded to cpu first, and eventually moved to gpu
        # (trick Mac M1 to use f16 tensors)
        if self.device != "cpu":
            self.clip_model = self.clip_model.to(self.device)

        valid_yolo_versions = ["yolov8x", "yolov5su"]
        if yolo_ver not in valid_yolo_versions:
            raise ValueError(f"Invalid YOLO version '{yolo_ver}'. Must be one of {valid_yolo_versions}.")

        self._embed_categories()

        print("[INFO] Initializing YoloClip pipeline")
        print(f"[INFO] YOLO version: {yolo_ver}")
        print("")

    def _encode_text(self, text):
        text_ = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_text(text_)

    def _encode_img(self, image):
        image_ = self.clip_prep(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_image(image_)

    def _embed_categories(self):
        for category_id in self.categories.keys():
            cur_category = self.categories[category_id]['category']
            with torch.no_grad():
                cur_category_enc = self._encode_text(f"a photo of {cur_category}")
            self.categories[category_id].update({"encoding": cur_category_enc})

    def __call__(self, img_sample, prompt, show=False, show_yolo=False, timeit=False):

        if timeit:
            start = time.time()

        # Use YOLO to find relevant objects

        img = img_sample.img

        yolo_results_ = self.yolo_model(img_sample.path, verbose=False)[0]
        yolo_results = yolo_results_.boxes.xyxy

        if not self.quiet:
            print(f"[INFO] YOLO found {yolo_results.shape[0]} objects")

        if yolo_results.shape[0] == 0:
            print(f"[WARN] YOLO ({self.yolo_ver}) couldn't find any object in {img_sample.path}!")
            return {"IoU": 0, "cosine": np.nan, "euclidean": np.nan, "dotproduct": np.nan, "grounding": np.nan}

        # Use CLIP to encode each relevant object image

        images_encs = list()

        for i in range(yolo_results.shape[0]):
            bbox = yolo_results[i, 0:4].cpu().numpy()

            sub_img = img.crop(bbox)

            with torch.no_grad():
                sub_img_enc = self._encode_img(sub_img)

            images_encs.append(sub_img_enc)

        images_encs = torch.cat(images_encs, dim=0)

        # Use CLIP to encode the text prompt
        prompt_enc = self._encode_text(prompt)

        # Compute distance metrics between found objects and the prompt

        # Dot product similarity
        d_sims = torch.mm(prompt_enc, images_encs.t()).squeeze()

        # Cosine similarity
        c_sims = cosine_similarity(prompt_enc, images_encs).squeeze()

        # Euclidean distance
        e_dists = torch.cdist(prompt_enc, images_encs, p=2).squeeze()

        best_idx = int(c_sims.argmax())

        # Save predicted bbox and true bbox
        pred_bbox = yolo_results[best_idx, 0:4].tolist()
        gt_bbox = img_sample.bbox

        # Compute IoU

        iou = box_iou(
            torch.tensor(pred_bbox).unsqueeze(0),
            torch.tensor(gt_bbox).unsqueeze(0)
        ).item()

        # Compute grounding accuracy

        pred_img = img.crop(pred_bbox)
        pred_img = self.clip_prep(pred_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_img_enc = self.clip_model.encode_image(pred_img).float()

        dotproducts, cosine_sims, euclidean_dists = dict(), dict(), dict()

        for category_id in self.categories.keys():
            cur_categ = self.categories[category_id]['category']
            cur_categ_enc = self.categories[category_id]['encoding'].float()
            cosine_sims[cur_categ] = cosine_similarity(pred_img_enc, cur_categ_enc)

        pred_category = max(cosine_sims, key=cosine_sims.get)

        grd_correct = 1 if pred_category == img_sample.category else 0

        if not self.quiet:
            print(f"[INFO] true: {img_sample.category} | predicted: {pred_category}")

        # Show objects found by YOLO, if requested
        if show_yolo:
            plt.imshow(yolo_results_.plot())
            plt.axis("off")
            plt.title("YOLO findings")

        # Show the final prediction, if requested
        if show:
            display_preds(img, prompt, pred_bbox, gt_bbox, model_name=f"{self.yolo_ver} + CLIP({self.clip_ver})")

        # Print execution time, if requested
        if timeit:
            end = time.time()
            print(f"[INFO] Time elapsed: {end - start:.2f}s")

        return {
            "IoU": float(iou),
            "cosine": float(c_sims.max()),
            "euclidean": float(e_dists.min()),
            "dotproduct": float(d_sims.max()),
            "grounding": float(grd_correct)
        }
