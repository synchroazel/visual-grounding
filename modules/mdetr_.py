import clip
import pandas as pd
import torch
import time
import torchvision.transforms as T
from PIL import Image
from torchmultimodal.models.mdetr.model import mdetr_for_phrase_grounding
from torchvision.ops import box_iou
from torchvision.ops.boxes import box_convert
from transformers import RobertaTokenizerFast

from modules.utilities import display_preds, cosine_similarity


def rescale_boxes(boxes, size):
    w, h = size
    b = box_convert(boxes, "cxcywh", "xyxy")
    b = b * torch.tensor([w, h, w, h], dtype=torch.float32)
    return b


class MDETRvg:

    def __init__(self, categories, device="cpu", quiet=True):
        super().__init__()
        self.MDETR = mdetr_for_phrase_grounding()
        self.MDETR.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://pytorch.s3.amazonaws.com/models/multimodal/mdetr/pretrained_resnet101_checkpoint.pth"
        )["model_ema"])
        self.RoBERTa = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.quiet = quiet
        self.device = device
        self.categories = categories
        self.img_preproc = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.clip_model, self.clip_prep = clip.load("RN101", device=device)

        self._embed_categories()

        print("[INFO] Initializing MDETR pipeline")

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

    def __call__(self, img_sample, prompt, show=True, timeit=False):

        if timeit:
            start = time.time()

        # Encode the prompt with RoBERTa

        img = Image.open(img_sample.path)

        enc_text = self.RoBERTa.batch_encode_plus([prompt], padding="longest", return_tensors="pt")

        # Preprocess the image and run MDETR on image and prompt

        img_transformed = self.img_preproc(img)

        with torch.no_grad():
            out = self.MDETR([img_transformed], enc_text["input_ids"]).model_output

        probs = 1 - out.pred_logits.softmax(-1)[0, :, -1]

        boxes_scaled = rescale_boxes(out.pred_boxes[0, :], img.size)

        mdetr_results = pd.DataFrame(boxes_scaled.squeeze().numpy().reshape(-1, 4))
        mdetr_results.columns = ["xmin", "ymin", "xmax", "ymax"]
        mdetr_results["prob"] = probs.numpy()

        mdetr_results = mdetr_results.sort_values(by=['prob'], ascending=False)

        pred_bbox = mdetr_results.iloc[0, :4].tolist()
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
