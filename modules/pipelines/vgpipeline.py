import copy

import clip
import torch
from torchvision.ops import box_iou

from utilities import cosine_similarity


class VisualGroundingPipeline:

    def __init__(self,
                 categories,
                 clip_ver="RN50",
                 device="cpu",
                 quiet=True):
        self.categories = copy.deepcopy(categories)
        self.clip_ver = clip_ver
        self.clip_model, self.clip_prep = clip.load(clip_ver, device="cpu")
        self.device = device
        self.quiet = quiet

        # model is loaded to cpu first, and eventually moved to gpu
        # (trick Mac M1 to use f16 tensors)
        if self.device != "cpu":
            self.clip_model = self.clip_model.to(self.device)

        self._embed_categories()

    def _encode_text(self, text):
        text_ = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_text(text_)

    def _encode_img(self, image):
        image_ = self.clip_prep(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_image(image_)

    @staticmethod
    def _IoU(pred_bbox, gt_bbox):
        iou = box_iou(
            torch.tensor(pred_bbox).unsqueeze(0),
            torch.tensor(gt_bbox).unsqueeze(0)
        ).item()

        return iou

    def _grounding_accuracy(self, img_sample, pred_image_enc):
        all_c_sims = dict()

        for category_id in self.categories.keys():
            cur_categ = self.categories[category_id]['category']
            cur_categ_enc = self.categories[category_id]['encoding'].float()

            all_c_sims[cur_categ] = cosine_similarity(pred_image_enc, cur_categ_enc)

        pred_category = max(all_c_sims, key=all_c_sims.get)

        # if not self.quiet:
        #     print(f"[INFO] true: {img_sample.category} | predicted: {pred_category}")

        return 1 if pred_category == img_sample.category else 0

    def _embed_categories(self):
        for category_id in self.categories.keys():
            cur_category = self.categories[category_id]['category']
            with torch.no_grad():
                cur_category_enc = self._encode_text(f"a photo of {cur_category}")
            self.categories[category_id].update({"encoding": cur_category_enc})

    def __call__(self, *args, **kwargs):
        return None
