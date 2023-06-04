import time

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
from tqdm import tqdm

from modules.utilities import cosine_similarity, display_preds, find_best_bbox, downsample_map
from modules.pipelines.vgpipeline import VisualGroundingPipeline


class ClipSeg(VisualGroundingPipeline):

    def __init__(self,
                 categories,
                 method,
                 n_segments,
                 clip_ver="ViT-L/14",
                 q=0.95,
                 d=16,
                 device="cpu",
                 quiet=False):

        VisualGroundingPipeline.__init__(self, categories, clip_ver, device, quiet)

        self.method = method
        self.n_segments = n_segments
        self.q = q
        self.d = d

        valid_methods = ["s", "w"]
        if self.method not in valid_methods:
            raise ValueError(f"Method `{method}` not supported. Supported methods are: {valid_methods}.")

        print("[INFO] Initializing ClipSeg pipeline")
        print(f"[INFO] Segmentation method: {method}")
        print(f"[INFO] Number of segments: {n_segments}")
        print(f"[INFO] Threshold q.tile for filtering: {q}")
        print(f"[INFO] Downsampling factor: {d}")
        print("")

    def _compute_hmap(self, img_sample, np_image, prompt, method, masks):

        # Make sure np_image is an image with shape (h, w, 3)
        if len(np_image.shape) > 3 or (len(np_image.shape) == 3 and np_image.shape[-1] != 3):
            np_image = np_image[:, :, 0]

        if len(np_image.shape) == 2:
            np_image = np.stack((np_image,) * 3, axis=-1)

        hmaps = list()

        prompt_enc = self._encode_text(prompt)

        for i, n in enumerate(masks):

            # Compute regions according to chosen method
            segments = None
            if method == "s":
                # SLIC segmentation algorithm ()
                segments = slic(np_image, n_segments=n, compactness=10, sigma=1)
            elif method == "w":
                # Watershed segmentation algorithm ()
                segments = watershed(sobel(rgb2gray(np_image)), markers=n, compactness=0.001)

            if segments is None:
                raise Exception("Segments are None. Is method different from 's' or 'w'? ")

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

    def __call__(self, img_sample, prompt, show=False, show_pipeline=False, show_masks=False, timeit=False):

        if timeit:
            start = time.time()

        """ Pipeline core """

        # Get sample image
        img = img_sample.img

        # Convert image to np array
        np_image = img_as_float(io.imread(img_sample.path))

        # Compute an heatmap of CLIP scores
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

        # Use CLIP to encode the text prompt
        prompt_enc = self._encode_text(prompt).float()

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
            methods = {"w": "Watershed", "s": "SLIC"}
            display_preds(img_sample.img, prompt, pred_bbox, img_sample.bbox,
                          f"{methods[self.method]} + CLIP ({self.clip_ver})")

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
