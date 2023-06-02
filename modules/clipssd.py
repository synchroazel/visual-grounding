import os
import time

import clip

import numpy as np
import torch

from torchvision.ops import box_iou
from tqdm import tqdm

from .utilities import cosine_similarity, display_preds


class ClipSSD:

    def __init__(self,
                 categories,
                 confidence_t=0.5,
                 device="cpu",
                 quiet=False):

        self.categories = categories
        self.clip_model, self.clip_prep = clip.load("ViT-L/14", device=device)
        self.confidence_t = confidence_t  #
        self.device = device
        self.quiet = quiet

        self.ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

        self.ssd_model.to(device)
        self.ssd_model.eval()

        for category_id in categories.keys():
            cur_category = categories[category_id]['category']
            with torch.no_grad():
                cur_category_enc = self._encode_text(f"a photo of {cur_category}")
            categories[category_id].update({"encoding": cur_category_enc})

        print(f"[INFO] Confidence treshold: {confidence_t}")

    def _encode_text(self, text):
        text_ = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_text(text_)

    def _encode_img(self, image):
        image_ = self.clip_prep(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_image(image_)

    def _propose(self, image_path, original_size):

        def resize_bbox(bbox, in_size, out_size):
            """Resize bounding boxes according to image resize.

            Args:
                bbox (~numpy.ndarray): See the table below.
                in_size (tuple): A tuple of length 2. The height and the width
                    of the image before resized.
                out_size (tuple): A tuple of length 2. The height and the width
                    of the image after resized.

            .. csv-table::
                :header: name, shape, dtype, format

                :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
                ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"

            Returns:
                ~numpy.ndarray:
                Bounding boxes rescaled according to the given image shapes.

            """
            bbox = bbox.copy()
            y_scale = float(out_size[0]) / in_size[0]
            x_scale = float(out_size[1]) / in_size[1]
            bbox[:, 0] = y_scale * bbox[:, 0]
            bbox[:, 2] = y_scale * bbox[:, 2]
            bbox[:, 1] = x_scale * bbox[:, 1]
            bbox[:, 3] = x_scale * bbox[:, 3]
            return bbox


        bboxes = []


        inputs = [self.utils.prepare_input(image_path)]
        tensor = self.utils.prepare_tensor(inputs)

        with torch.no_grad():
            detections_batch = self.ssd_model(tensor)

        results_per_input = self.utils.decode_results(detections_batch)
        best_results_per_input = [self.utils.pick_best(results, self.confidence_t) for results in results_per_input]

        bbox, _, _ = best_results_per_input[0]
        bbox *= 300
        bbox = resize_bbox(bbox, (300,300),original_size)
        bboxes.append(bbox)
        return np.float32(bboxes[0]).tolist()

    def __call__(self, img_sample, prompt, show=False,
                 timeit=False):

        if timeit:
            start = time.time()

        # Convert image to np array
        image_path = img_sample.path
        img = img_sample.img
        bboxes = self._propose(image_path,(img_sample.shape[1],img_sample.shape[2]))

        if len(bboxes) == 0:
            return {"IoU": 0, "cosine": np.nan, "euclidean": np.nan, "dotproduct": np.nan, "grounding": np.nan}

        # Use CLIP to encode each relevant object image

        images_encs = list()

        for bbox in bboxes:
            # bbox = yolo_results[i, 0:4].cpu().numpy()

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
        e_dists = torch.cdist(prompt_enc.float(), images_encs.float(), p=2).squeeze()

        best_idx = int(c_sims.argmax())

        # Save predicted bbox and true bbox
        pred_bbox = bboxes[best_idx]
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

        # Show the final prediction, if requested
        if show:
            display_preds(img, prompt, pred_bbox, gt_bbox, model_name="SSD+CLIP")

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
