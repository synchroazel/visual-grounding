import time

import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.segmentation import slic, watershed
from skimage.util import img_as_float
from torchvision.ops import box_iou
from tqdm import tqdm

from modules.utilities import cosine_similarity, display_preds, find_best_bbox, downsample_map


class ClipSeg:

    def __init__(self,
                 categories,
                 method,
                 n_segments=None,
                 q=0.95,
                 d=16,
                 device="cpu",
                 quiet=False):

        self.categories = categories
        self.method = method
        self.clip_model, self.clip_prep = clip.load("ViT-L/14", device=device)
        self.n_segments = n_segments
        self.q = q
        self.d = d
        self.device = device
        self.quiet = quiet

        valid_methods = ["s", "w"]
        if self.method not in valid_methods:
            raise ValueError(f"Method `{method}` not supported. Supported methods are: {valid_methods}.")

        for category_id in categories.keys():
            cur_category = categories[category_id]['category']
            with torch.no_grad():
                cur_category_enc = self._encode_text(f"a photo of {cur_category}")
            categories[category_id].update({"encoding": cur_category_enc})

        print(f"[INFO] Segmentation method: {method}")
        print(f"[INFO] Number of segments: {n_segments}")
        print(f"[INFO] Threshold q.tile for filtering: {q}")
        print(f"[INFO] Downsampling factor: {d}")

    def _encode_text(self, text):
        text_ = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_text(text_)

    def _encode_img(self, image):
        image_ = self.clip_prep(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_image(image_)

    def _compute_hmap(self, img_sample, np_image, prompt, method, masks):

        hmaps = list()

        prompt_enc = self._encode_text(prompt)

        for i, n in enumerate(masks):

            # Compute regions according to chosen method

            if method == "s":
                # SLIC segmentation algorithm ()
                segments = slic(np_image, n_segments=n, compactness=10, sigma=1)
            elif method == "w":
                # Watershed segmentation algorithm ()
                segments = watershed(sobel(rgb2gray(np_image)), markers=n, compactness=0.001)

            regions = regionprops(segments)

            if len(regions) == 1:
                # If the algo returned only 1 region, skip this iteration
                # (may happen, with low-segments masks)
                continue

            # Compute CLIP encodings for each region

            images_encs = list()

            regions = tqdm(regions, desc=f"[INFO] Computing CLIP masks", leave=False) if not self.quiet else regions

            for region in regions:
                rect = region.bbox
                rect = (rect[1], rect[0], rect[3], rect[2])

                sub_image = img_sample.img.crop(rect)
                image_enc = self._encode_img(sub_image)
                images_encs.append(image_enc)

            # Assign a score to each region according to prompt similarity (creating a heatmap)

            images_encs = torch.cat(images_encs, dim=0)
            scores = prompt_enc @ images_encs.T
            scores = scores.squeeze().cpu().numpy()
            heatmap = np.zeros((segments.shape[0], segments.shape[1]))

            for i in range(segments.shape[0]):
                for j in range(segments.shape[1]):
                    heatmap[i, j] = scores[segments[i, j] - 1]

            hmaps.append(heatmap)

        # Finally, return the pooled heatmap and the list of all heatmaps computed

        pmap = np.mean(np.array(hmaps), axis=0)

        return pmap, hmaps

    def __call__(self, img_sample, prompt, show=False, show_pipeline=False, show_masks=False,
                 timeit=False):

        if timeit:
            start = time.time()

        # Convert image to np array
        np_image = img_as_float(io.imread(img_sample.path))

        p_heatmap, heatmaps = self._compute_hmap(img_sample, np_image, prompt, self.method, self.n_segments)

        # Shut down pixels below a certain threshold
        ths = np.quantile(p_heatmap.flatten(), self.q)
        fp_heatmap = p_heatmap.copy()
        fp_heatmap[p_heatmap < ths] = ths

        # Downsample the heatmap by a factor d
        dfp_heatmap = downsample_map(fp_heatmap, self.d)

        # Find the best bounding box
        pred_bbox = find_best_bbox(dfp_heatmap, lower_bound=-0.75)

        if pred_bbox is None:
            return {"IoU": 0, "cosine": np.nan, "euclidean": np.nan, "dotproduct": np.nan, "grounding": np.nan}

        if self.d > 1:
            pred_bbox = [pred_bbox[0] * self.d + self.d // 2,
                         pred_bbox[1] * self.d + self.d // 2,
                         pred_bbox[2] * self.d - self.d // 2,
                         pred_bbox[3] * self.d - self.d // 2]

        # Convert bbox format
        pred_bbox = [pred_bbox[0], pred_bbox[1], pred_bbox[2] + pred_bbox[0], pred_bbox[3] + pred_bbox[1]]

        # Get ground truth bbox
        gt_bbox = img_sample.bbox

        # Compute IoU
        iou = box_iou(
            torch.tensor(pred_bbox).unsqueeze(0),
            torch.tensor(gt_bbox).unsqueeze(0)
        ).item()

        # Compute distance metrics
        pred_img = img_sample.img.crop(gt_bbox)
        pred_image_enc = self._encode_img(pred_img).float()
        prompt_enc = self._encode_text(prompt).float()

        cosine_sim = cosine_similarity(prompt_enc, pred_image_enc)
        euclidean_dist = torch.cdist(prompt_enc, pred_image_enc, p=2).squeeze()
        dotproduct = prompt_enc @ pred_image_enc.T

        # Compute grounding accuracy

        all_c_sims = dict()

        for category_id in self.categories.keys():
            cur_categ = self.categories[category_id]['category']
            cur_categ_enc = self.categories[category_id]['encoding'].float()

            all_c_sims[cur_categ] = cosine_similarity(pred_image_enc, cur_categ_enc)

        pred_category = max(all_c_sims, key=all_c_sims.get)

        grd_correct = 1 if pred_category == img_sample.category else 0

        # Show all masks, if requested
        if show_masks:
            fig, axes = plt.subplots(1, len(heatmaps), figsize=(20, 5))
            for i, heatmap in enumerate(heatmaps):

                for ax in axes.ravel():
                    ax.axis("off")

                axes[i].imshow(np_image, alpha=0.25)
                axes[i].imshow(heatmap, alpha=0.75)
                axes[i].set_title(f"#{i + 1}")

        # Show the mask processing pipeline, if requested
        if show_pipeline:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            for ax in axes.ravel():
                ax.axis("off")

            axes[0].imshow(np_image)
            axes[0].set_title("original image")

            axes[1].imshow(np_image, alpha=0.25)
            axes[1].imshow(p_heatmap, alpha=0.75)
            axes[1].set_title("pooled heatmap")

            axes[2].imshow(np_image, alpha=0.25)
            axes[2].imshow(fp_heatmap, alpha=0.75)
            axes[2].set_title("filtered heatmap")

            axes[3].imshow(np_image, alpha=0.25)
            w, h = np_image.shape[1], np_image.shape[0]
            dfp_heatmap_ = cv2.resize(dfp_heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
            axes[3].imshow(dfp_heatmap_, alpha=0.75)
            axes[3].set_title("dsampled heatmap")

        # Show the final prediction, if requested
        if show:
            display_preds(img_sample.img, prompt, pred_bbox, img_sample.bbox, f"{''.join(self.method).upper()} + CLIP")

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
