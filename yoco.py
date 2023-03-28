import clip
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image


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
            results.show()

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

    def __call__(self, img_path, prompt):

        if not self.quiet:
            print("[INFO] Running YOLO on the image...")

        yolo_results = self.yolo_model(img_path)

        if not self.quiet:
            print(f"[INFO] YOLO found {yolo_results.shape[0]} objects")

        img = Image.open(img_path)

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
            best_idx = int(e_dists.argmin())
        else:
            best_idx = int(c_sims.argmin())

        # TODO: implement Intersection over Union (IoU)

        # TODO: implement Recall

        if not self.quiet:
            # Display the image with the bbox for the object chosen
            ans = yolo_results.iloc[best_idx]

            best_bbox = ans[0:4].tolist()

            rect = patches.Rectangle(
                (best_bbox[0], best_bbox[1]), best_bbox[2] - best_bbox[0], best_bbox[3] - best_bbox[1],
                linewidth=1.5, edgecolor=(0, 1, 0), facecolor='none'
            )

            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.add_patch(rect)
            ax.axis("off")
            plt.title(f"\"{prompt.capitalize()}\"")
            plt.show()

        return {"cosine": float(c_sims.min()), "euclidean": float(e_dists.min())}
