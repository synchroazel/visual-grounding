import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from modules.utilities import cosine_similarity, display_preds
from modules.vgpipeline import VisualGroundingPipeline


class YoloClip(VisualGroundingPipeline):

    def __init__(self,
                 categories,
                 yolo_ver="yolov8x",
                 clip_ver="RN50",
                 device="cpu",
                 quiet=False):

        VisualGroundingPipeline.__init__(self, categories, clip_ver, device, quiet)

        self.yolo_ver = yolo_ver
        self.yolo_model = YOLO(self.yolo_ver + ".pt")

        valid_yolo_versions = ["yolov8x", "yolov5su"]
        if yolo_ver not in valid_yolo_versions:
            raise ValueError(f"Invalid YOLO version '{yolo_ver}'. Must be one of {valid_yolo_versions}.")

        print("[INFO] Initializing YoloClip pipeline")
        print(f"[INFO] YOLO version: {yolo_ver}")
        print("")

    def __call__(self, img_sample, prompt, show=False, show_yolo=False, timeit=False):

        if timeit:
            start = time.time()

        # Get sample image
        img = img_sample.img

        # Use YOLO to propose relevant objects
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

        # Compute the best bbox according to cosine similarity
        c_sims = cosine_similarity(prompt_enc, images_encs).squeeze()
        best_idx = int(c_sims.argmax())

        # Get best bbox
        pred_bbox = yolo_results[best_idx, 0:4].tolist()

        # Crop around the best bbox and encode
        pred_image = img.crop(pred_bbox)
        pred_image_enc = self._encode_img(pred_image)

        # Get ground truth bbox
        gt_bbox = img_sample.bbox

        """ Metrics computation """

        # Compute IoU
        iou = self._IoU(pred_bbox, gt_bbox)

        # Compute grounding accuracy
        grd_correct = self._grounding_accuracy(img_sample, pred_image_enc)

        # Compute distance metrics
        dotproduct = prompt_enc @ pred_image_enc.T  # dot product
        cosine_sim = cosine_similarity(prompt_enc, pred_image_enc)  # cosine similarity
        euclidean_dist = torch.cdist(prompt_enc, pred_image_enc, p=2).squeeze()  # euclidean distance

        """ Display results """

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
            "cosine": float(cosine_sim),
            "euclidean": float(euclidean_dist),
            "dotproduct": float(dotproduct),
            "grounding": float(grd_correct),
        }
