import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, DetrForObjectDetection
from transformers.utils import logging

from modules.utilities import cosine_similarity, display_preds
from modules.vgpipeline import VisualGroundingPipeline


class DetrClip(VisualGroundingPipeline):
    def __init__(self,
                 categories,
                 clip_ver="RN50",
                 device="cpu",
                 quiet=False):

        logging.set_verbosity_error()

        VisualGroundingPipeline.__init__(self, categories, clip_ver, device, quiet)

        self.image_prep = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        print("[INFO] Initializing DetrClip pipeline")
        print("")

    def __call__(self, img_sample, prompt, show=False, show_detr=False, timeit=False):

        if timeit:
            start = time.time()

        """ Pipeline core """

        # Get sample image
        img = img_sample.img

        # Make sure image has shape (h, w, 3)
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

        # Find the best object according to cosine similarity
        c_sims = cosine_similarity(prompt_enc, images_encs).squeeze()
        best_idx = int(c_sims.argmax())

        # Get best bbox
        pred_bbox = detr_results[best_idx, 0:4].tolist()

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

        # Show objects found by DETR, if requested
        if show_detr:
            fig, ax = plt.subplots()
            ax.imshow(img)
            for i in range(detr_results.shape[0]):
                bbox = results['boxes'][i].cpu().numpy()
                # print(bbox)
                rect = plt.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    linewidth=2, edgecolor=(0, 1, 0), facecolor="none"
                )
                ax.add_patch(rect)
            ax.axis("off")
            plt.title("DETR findings")
            plt.show()

        # Show the final prediction, if requested
        if show:
            display_preds(img, prompt, pred_bbox, gt_bbox, model_name=f"DETR + CLIP ({self.clip_ver})")

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
