import clip
import torch
from PIL import Image
from torchvision.ops import box_iou

from vgutils import display_preds


class YOLOv5:
    """
    Wrapper around YOLOv5 to perform object detection.

    """

    def __init__(self, yolo_ver="yolov5s", device="cpu", quiet=True):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', yolo_ver, pretrained=True, device=device)
        self.quiet = quiet

    def __call__(self, img_path):
        results = self.yolo_model(img_path)

        if not self.quiet:
            # results.show()
            pass

        results = results.pandas().xyxy[0]

        return results


class CLIP:
    """
    Wrapper around CLIP to extract image or text features.

    """

    def __init__(self, clip_ver="RN50", device="cpu", quiet=True):
        self.clip_model, self.preprocess = clip.load(clip_ver, device=device)
        self.quiet = quiet
        self.device = device

    def __call__(self, content):

        if isinstance(content, str):
            text = clip.tokenize(content).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.encode_text(text)

            return text_features

        elif isinstance(content, Image.Image):
            sub_img = self.preprocess(content).unsqueeze(0).to(self.device)

            with torch.no_grad():
                imgs_features = self.clip_model.encode_image(sub_img)

            return imgs_features

        else:
            raise TypeError("Content must be a string or a PIL image!")


class YoloClip:
    """
    A wrapper around YOLO and CLIP to perform visual grounding given an image and a prompt.

    """

    def __init__(self, categories, yolo_ver="yolov5s", clip_ver="RN50", device="cpu", quiet=True,
                 dist_metric="euclidean"):
        self.yolo_model = YOLOv5(yolo_ver, quiet=quiet)
        self.clip_model = CLIP(clip_ver, quiet=quiet)
        self.quiet = quiet
        self.device = device
        self.categories = categories

        for category_id in categories.keys():
            cur_category = categories[category_id]['category']
            cur_category_enc = self.clip_model(f"a picture of {cur_category}")
            categories[category_id].update({"encoding": cur_category_enc})

        # TODO: maybe add more metrics for the distance?

        valid_metrics = ["euclidean", "cosine", "dotproduct"]
        if dist_metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{dist_metric}'. Must be one of {valid_metrics}.")

        self.dist_metric = dist_metric

    def __call__(self, img_sample, prompt, show=True):

        # 1. Use YOLO to find relevant objects

        img = Image.open(img_sample.path)

        if not self.quiet:
            print("[INFO] Running YOLO on the image...")

        yolo_results = self.yolo_model(img_sample.path)

        if not self.quiet:
            print(f"[INFO] YOLO found {yolo_results.shape[0]} objects")

        # TODO: In case YOLO doesn't find any object, what should we do?  [Do not consider sample]

        if yolo_results.shape[0] == 0:
            raise ValueError("YOLO didn't find any object in the image!")

        # 2.1 Use CLIP to encode each relevant object image

        imgs_encoding = list()

        if not self.quiet:
            print("[INFO] Running CLIP on detected objects...")

        for i in range(yolo_results.shape[0]):
            bbox = yolo_results.iloc[i, 0:4]

            sub_img = img.crop(bbox)

            imgs_encoding.append(self.clip_model(sub_img))

        imgs_encoding = torch.cat(imgs_encoding, dim=0)

        # 2.2 Use CLIP to encode the text prompt

        if not self.quiet:
            print("[INFO] Running CLIP on the prompt...")

        text_encoding = self.clip_model(prompt)

        # yolo_results["prob"] = torch.mm(text_encoding, imgs_encoding.t()).squeeze()

        # 3.1 Compute distance metrics and find the best object according to one of them

        pred_bbox_idx = dict()

        # Dot product similarity
        d_sims = torch.mm(text_encoding, imgs_encoding.t()).squeeze()

        # Cosine similarity
        c_sims = torch.nn.functional.cosine_similarity(text_encoding, imgs_encoding, dim=1).squeeze()

        # Euclidean distance
        e_dists = torch.cdist(text_encoding, imgs_encoding, p=2).squeeze()

        # Minkowski distance
        # TODO: implement

        pred_bbox_idx["dotproduct"] = int(d_sims.argmax())
        pred_bbox_idx["cosine"] = int(c_sims.argmax())
        pred_bbox_idx["euclidean"] = int(e_dists.argmin())

        # index of the best bounding box according to user-chosen metric
        best_idx = pred_bbox_idx[self.dist_metric]

        # 3.2 Save predicted bbox and true bbox

        pred_bbox = yolo_results.iloc[best_idx, 0:4].tolist()

        gt_bbox = img_sample.bbox

        # 4.1 Compute other metrics: Intersection over Union (IoU)

        iou = box_iou(
            torch.tensor([pred_bbox]),
            torch.tensor([gt_bbox])
        )

        # 4.2 Compute other metrics: visual grounding accuracy

        pred_img = img.crop(pred_bbox)
        pred_img_enc = self.clip_model(pred_img)

        pred_bbox_categ = dict()

        all_d_sims, all_c_sims, all_e_dists = dict(), dict(), dict()

        for category_id in self.categories.keys():
            cur_categ = self.categories[category_id]['category']
            cur_categ_enc = self.categories[category_id]['encoding']
            all_d_sims[cur_categ] = torch.mm(pred_img_enc, cur_categ_enc.t()).squeeze()
            all_c_sims[cur_categ] = torch.nn.functional.cosine_similarity(pred_img_enc, cur_categ_enc, dim=1).squeeze()
            all_e_dists[cur_categ] = torch.cdist(pred_img_enc, cur_categ_enc, p=2).squeeze()

        pred_bbox_categ["dotproduct"] = max(all_d_sims, key=all_d_sims.get)
        pred_bbox_categ["cosine"] = max(all_c_sims, key=all_c_sims.get)
        pred_bbox_categ["euclidean"] = min(all_e_dists, key=all_e_dists.get)

        # category of the best bounding box according to chosen metric
        pred_category = pred_bbox_categ[self.dist_metric]

        # If the category with the highest cosine similarity is the same
        # as the ground truth category, then the grounding is correct

        grd_correct = 1 if pred_category == img_sample.category else 0

        if not self.quiet:
            print(f"[INFO] true: {img_sample.category} | predicted: {pred_category}")

        # 5. Display results and return metrics

        if show:
            display_preds(img, prompt, pred_bbox, gt_bbox, model_name="YOLO+CLIP")

        return {
            "cosine": float(c_sims.min()),
            "euclidean": float(e_dists.min()),
            "dotproduct": float(d_sims.max()),
            "iou": float(iou),
            "recall": float(grd_correct)
        }
