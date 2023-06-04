import copy
import time

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import patches
from torchvision.ops import box_iou
from transformers import AutoImageProcessor, DetrForObjectDetection
from transformers.utils import logging

from modules.utilities import cosine_similarity, display_preds


class DetrClip:

    def __init__(self,
                 categories,
                 clip_ver="RN50",
                 device="cpu",
                 quiet=False):

        logging.set_verbosity_error()

        self.categories = copy.deepcopy(categories)
        self.image_prep = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.clip_ver = clip_ver
        self.clip_model, self.clip_prep = clip.load(clip_ver, device="cpu")
        self.device = device
        self.quiet = quiet

        # model is loaded to cpu first, and eventually moved to gpu
        # (trick Mac M1 to use f16 tensors)
        if self.device != "cpu":
            self.clip_model = self.clip_model.to(self.device)

        self._embed_categories()

        print("[INFO] Initializing DetrClip pipeline")
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

    def __call__(self, img_sample, prompt, show=False, show_detr=False, timeit=False):

        if timeit:
            start = time.time()

        img = img_sample.img

        # Make sure np_image is an image with shape (h, w, 3)

        np_image = np.array(img)

        if len(np_image.shape) > 3 or (len(np_image.shape) == 3 and np_image.shape[-1] != 3):
            np_image = np_image[:, :, 0]

        if len(np_image.shape) == 2:
            np_image = np.stack((np_image,) * 3, axis=-1)

        img = Image.fromarray(np_image)

        # Use DETR to find relevant objects

        inputs = self.image_prep(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = self.detr(**inputs)

        target_sizes = torch.tensor([img_sample.img.size[::-1]])
        results = self.image_prep.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        detr_results = results['boxes']

        # Use CLIP to encode each relevant object image

        images_encs = list()

        for i in range(detr_results.shape[0]):
            bbox = results['boxes'][i, 0:4].cpu().numpy()

            sub_img = img_sample.img.crop(bbox)

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
        pred_bbox = detr_results[best_idx, 0:4].tolist()
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

        # Show objects found by DETR, if requested
        if show_detr:
            plt.imshow(img)
            for i in range(detr_results.shape[0]):
                bbox = results['boxes'][i, 0:4].cpu().numpy()
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, facecolor="none"
                )
                plt.gca().add_patch(rect)
            plt.axis("off")
            plt.title("DETR findings")

        # Show the final prediction, if requested
        if show:
            display_preds(img, prompt, pred_bbox, gt_bbox, model_name=f"DETR + CLIP ({self.clip_ver})")

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
