import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchmultimodal.models.mdetr.model import mdetr_for_phrase_grounding
from torchvision.ops import box_iou
from torchvision.ops.boxes import box_convert
from transformers import RobertaTokenizerFast

from vgutils import display_preds

medtr_cpt_url = "https://pytorch.s3.amazonaws.com/models/multimodal/mdetr/pretrained_resnet101_checkpoint.pth"


def rescale_boxes(boxes, size):
    """
    Util to rescale predicted boxes to match image size

    """

    w, h = size
    b = box_convert(boxes, "cxcywh", "xyxy")
    b = b * torch.tensor([w, h, w, h], dtype=torch.float32)
    return b


class MDETRvg():
    """
    Wrapper around MEDTR to perform phrase grounding.

    """

    def __init__(self, categories, device="cpu", quiet=True):
        super().__init__()
        self.MDETR = mdetr_for_phrase_grounding()
        self.MDETR.load_state_dict(torch.hub.load_state_dict_from_url(medtr_cpt_url)["model_ema"])
        self.RoBERTa = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.quiet = quiet
        self.device = device
        self.categories = categories
        self.img_preproc = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img_sample, prompt, show=True):
        # 1. Encode the prompt with RoBERTa

        img = Image.open(img_sample.path)

        enc_text = self.RoBERTa.batch_encode_plus([prompt], padding="longest", return_tensors="pt")

        # 2. Preprocess the image and run MDETR on image and prompt

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
        best_prob = mdetr_results.iloc[0, 4]
        gt_bbox = img_sample.bbox

        # # Extract the text spans predicted by each box
        # positive_tokens = (out.pred_logits[0, keep].softmax(-1) > 0.1).nonzero().tolist()
        # predicted_spans = defaultdict(str)
        # for tok in positive_tokens:
        #     item, pos = tok
        #     if pos < 255:
        #         span = enc_text.token_to_chars(0, pos)
        #         predicted_spans[item] += " " + prompt[span.start:span.end]
        # boxes_scaled = [boxes_scaled[int(k)] for k in sorted(list(predicted_spans.keys()))]
        # labels = [predicted_spans[k] for k in sorted(list(predicted_spans.keys()))]

        # Compute Intersection over Union (IoU)

        iou = box_iou(
            torch.tensor([pred_bbox]),
            torch.tensor([gt_bbox])
        )

        # Compute Precision@k (k = 0.5, 0.7, 0.9)

        prec = list()

        for k in [0.5, 0.7, 0.9]:
            if best_prob > k:
                prec.append(1)
            else:
                prec.append(0)

        # Display results and return metrics

        if show:
            display_preds(img, prompt, pred_bbox, gt_bbox, model_name="MDETR")

        return {
            "iou": float(iou),
            "pr@k": prec
        }
