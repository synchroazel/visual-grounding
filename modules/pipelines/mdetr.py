import time

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchmultimodal.models.mdetr.model import mdetr_for_phrase_grounding
from torchvision.ops.boxes import box_convert
from transformers import RobertaTokenizerFast

from modules.pipelines.vgpipeline import VisualGroundingPipeline
from modules.utilities import display_preds, cosine_similarity


class MDETRvg(VisualGroundingPipeline):

    def __init__(self,
                 categories,
                 clip_ver="RN101",
                 device="cpu",
                 quiet=True):

        VisualGroundingPipeline.__init__(self, categories, clip_ver, device, quiet)

        cpt_url = "https://pytorch.s3.amazonaws.com/models/multimodal/mdetr/pretrained_resnet101_checkpoint.pth"

        self.MDETR = mdetr_for_phrase_grounding()
        self.MDETR.load_state_dict(torch.hub.load_state_dict_from_url(cpt_url)["model_ema"])
        self.RoBERTa = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.img_preproc = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print("[INFO] Initializing MDETR pipeline")

    @staticmethod
    def rescale_boxes(boxes, size):
        w, h = size
        b = box_convert(boxes, "cxcywh", "xyxy")
        b = b * torch.tensor([w, h, w, h], dtype=torch.float32)
        return b

    def __call__(self, img_sample, prompt, show=True, timeit=False):

        if timeit:
            start = time.time()

        """ Pipeline core """

        # Get sample image
        img = Image.open(img_sample.path)

        # Encode the prompt with RoBERTa
        enc_text = self.RoBERTa.batch_encode_plus([prompt], padding="longest", return_tensors="pt")

        # Preprocess the image for MDETR
        img_transformed = self.img_preproc(img)

        # Run MDETR on image and prompt
        with torch.no_grad():
            out = self.MDETR([img_transformed], enc_text["input_ids"]).model_output

        # Parse MDETR results to get detections bboxes and probabilities
        probs = 1 - out.pred_logits.softmax(-1)[0, :, -1]
        boxes_scaled = self.rescale_boxes(out.pred_boxes[0, :], img.size)
        mdetr_results = pd.DataFrame(boxes_scaled.squeeze().numpy().reshape(-1, 4))
        mdetr_results.columns = ["xmin", "ymin", "xmax", "ymax"]
        mdetr_results["prob"] = probs.numpy()
        mdetr_results = mdetr_results.sort_values(by=['prob'], ascending=False)

        # Get best bbox
        pred_bbox = mdetr_results.iloc[0, :4].tolist()

        # Use CLIP to encode the prompt
        prompt_enc = self._encode_text(prompt)

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
        cosine_sim = cosine_similarity(prompt_enc, pred_image_enc)
        euclidean_dist = torch.cdist(prompt_enc, pred_image_enc, p=2).squeeze()
        dotproduct = prompt_enc @ pred_image_enc.T

        """ Display results """

        # Show the final prediction, if requested
        if show:
            display_preds(img, prompt, pred_bbox, gt_bbox, model_name="MDETR")

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
