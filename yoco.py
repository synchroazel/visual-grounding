import clip
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou


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


class YOCO:
    """
    You Only Clip Once.

    A wrapper around YOLO and CLIP to perform visual grounding given an image and a prompt.

    """

    def __init__(self, yolo_ver="yolov5s", clip_ver="RN50", device="cpu", quiet=True, dist_metric="euclidean"):
        self.yolo_model = YOLOv5(yolo_ver, quiet=quiet)
        self.clip_model = CLIP(clip_ver, quiet=quiet)
        self.quiet = quiet
        self.device = device

        # TODO: maybe add more metrics for the distance?

        valid_metrics = ["euclidean", "cosine"]
        if dist_metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{dist_metric}'. Must be one of {valid_metrics}.")

        self.dist_metric = dist_metric

    def __call__(self, img_sample, prompt):

        if not self.quiet:
            print("[INFO] Running YOLO on the image...")

        yolo_results = self.yolo_model(img_sample.path)

        if not self.quiet:
            print(f"[INFO] YOLO found {yolo_results.shape[0]} objects")

        # In case YOLO doesn't find any object, what should we do?

        if yolo_results.shape[0] == 0:
            return {
                "cosine": np.nan,  # what should we do?
                "euclidean": np.nan,  # what should we do?
                "iou": 0.0,
                "recall": 0.0
            }

        img = Image.open(img_sample.path)

        imgs_encoding = list()

        if not self.quiet:
            print("[INFO] Running CLIP on detected objects...")

        for i in range(yolo_results.shape[0]):
            bbox = yolo_results.iloc[i, 0:4]

            sub_img = img.crop(bbox)

            imgs_encoding.append(self.clip_model(sub_img))

        imgs_encoding = torch.cat(imgs_encoding, dim=0)

        if not self.quiet:
            print("[INFO] Running CLIP on the prompt...")

        text_encoding = self.clip_model(prompt)

        # Compute the Cosine similarity between the prompt and each object
        c_sims = torch.nn.functional.cosine_similarity(text_encoding, imgs_encoding, dim=1).squeeze()

        # Compute the Euclidean distance between the prompt and each object
        e_dists = torch.cdist(text_encoding, imgs_encoding, p=2).squeeze()

        if self.dist_metric == "euclidean":
            dists = e_dists
        else:
            dists = c_sims

        # Find the object with the minimum distance to the prompt

        best_idx = int(dists.argmin())

        sorted_idxs = np.argsort(dists)

        obj = yolo_results.iloc[best_idx]

        pred_bbox = obj[0:4].tolist()

        gt_bbox = img_sample.bbox

        gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]

        # Compute the Intersection over Union (IoU) between the predicted and ground-truth bboxes

        tensor_pred_bbox = torch.tensor([pred_bbox])
        tensor_gt_bbox = torch.tensor([gt_bbox])

        iou = box_iou(tensor_pred_bbox, tensor_gt_bbox)

        ### UNSURE ABOUT THIS PART ###

        # Compute the Recall@K metric
        # MAYBE: https://ar5iv.labs.arxiv.org/html/2206.15462 ??

        k = 3

        if yolo_results.shape[0] < k:
            k = yolo_results.shape[0]

        recall_at_k = 0

        for i in range(k):

            cur_obj = yolo_results.iloc[int(sorted_idxs[i])]

            pred_bbox = cur_obj[0:4].tolist()

            gt_bbox = img_sample.bbox

            gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]

            tensor_pred_bbox = torch.tensor([pred_bbox])
            tensor_gt_bbox = torch.tensor([gt_bbox])

            iou = box_iou(tensor_pred_bbox, tensor_gt_bbox)

            if iou > 0.5:
                recall_at_k = 1
                break

        ### END  ###

        # TODO: check if recall implementation is correct

        if not self.quiet:
            # Display the image with the bbox for the object chosen

            pred_rect = patches.Rectangle(
                (pred_bbox[0], pred_bbox[1]), pred_bbox[2] - pred_bbox[0], pred_bbox[3] - pred_bbox[1],
                linewidth=1.5, edgecolor=(0, 1, 0), facecolor='none'
            )

            gt_rect = patches.Rectangle(
                (gt_bbox[0], gt_bbox[1]), gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1],
                linewidth=1.5, edgecolor=(1, 0, 0), facecolor='none'
            )

            fig, ax = plt.subplots()
            ax.imshow(img)

            ax.add_patch(pred_rect)
            ax.text(pred_bbox[0], pred_bbox[1], "predicted", color=(1, 1, 1),
                    bbox={"facecolor": (0, 1, 0), "pad": 2, "color": (0, 1, 0)})

            ax.add_patch(gt_rect)
            ax.text(gt_bbox[0], gt_bbox[3], "true", color=(1, 1, 1),
                    bbox={"facecolor": (1, 0, 0), "pad": 2, "color": (1, 0, 0)})

            ax.axis("off")
            plt.title(f"\"{prompt.capitalize()}\"\n")
            plt.show()

            plt.close(fig)

        return {
            "cosine": float(c_sims.min()),
            "euclidean": float(e_dists.min()),
            "iou": float(iou),
            "recall": recall_at_k
        }
