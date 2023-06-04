import time

import numpy as np
import torch

from modules.utilities import cosine_similarity, display_preds
from modules.vgpipeline import VisualGroundingPipeline


class ClipSSD(VisualGroundingPipeline):

    def __new__(cls, *args, **kwargs):

        # Single Shot Detector (SSD) requires CUDA.
        # Check if the selected device if CUDA before instantiating the class.

        if kwargs["device"] != torch.device("cuda"):
            print("[ERROR] Single Shot Detector requires CUDA. Returning empty object.")
            print("")
            return VisualGroundingPipeline.__new__(VisualGroundingPipeline)
        else:
            return super(ClipSSD, cls).__new__(cls, *args, **kwargs)

    def __init__(self,
                 categories,
                 confidence_t=0.5,
                 clip_ver="ViT-L/14",
                 device="cpu",
                 quiet=False):

        VisualGroundingPipeline.__init__(self, categories, clip_ver, device, quiet)

        self.confidence_t = confidence_t

        self.ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

        self.ssd_model.to(device)
        self.ssd_model.eval()

        print("[INFO] Initializing ClipSSD pipeline")
        print(f"[INFO] Confidence treshold: {confidence_t}")
        print("")

    def _propose(self, image_path, original_size):

        def _resize_bbox(bbox, in_size, out_size):
            """
            Resize bounding boxes according to image resize.

            Args:
                bbox: (np.ndarray) bounding boxes of (y_min, x_min, y_max, x_max)
                in_size: (tuple) the height and the width of the image before resized
                out_size: (tuple) The height and the width of the image after resized
            Returns:
                (np.ndarray) bounding boxes rescaled according to the given image shapes

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
        bbox = _resize_bbox(bbox, (300, 300), original_size)
        bboxes.append(bbox)

        return np.float32(bboxes[0]).tolist()

    def __call__(self, img_sample, prompt, show=False, timeit=False):

        if timeit:
            start = time.time()

        """ Pipeline core """

        # Get sample image
        image_path = img_sample.path
        img = img_sample.img

        # Use SSD to propose relevant objects
        bboxes = self._propose(image_path, (img_sample.shape[1], img_sample.shape[2]))

        # Handle case where no object is proposed
        if len(bboxes) == 0:
            return {"IoU": 0, "cosine": np.nan, "euclidean": np.nan, "dotproduct": np.nan, "grounding": np.nan}

        # Use CLIP to encode each relevant object detected
        images_encs = list()
        for bbox in bboxes:
            sub_img = img.crop(bbox)
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
        pred_bbox = bboxes[best_idx]

        # Use CLIP to encode the prompt
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

        # Show the final prediction, if requested
        if show:
            display_preds(img, prompt, pred_bbox, gt_bbox, model_name="SSD+CLIP")

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
