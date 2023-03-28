import clip
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image


class YOLOv5:
    """
    Wrapper around YOLOv5 to perform object detection.

    """

    def __init__(self, yolo_ver="yolov5s", device="cpu", debug=False):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', yolo_ver, pretrained=True, device=device)
        self.debug = debug

    def __call__(self, img_path):
        results = self.yolo_model(img_path)

        if self.debug:
            results.show()

        results = results.pandas().xyxy[0]

        if self.debug:
            print(f"Found {results.shape[0]} objects")

        return results


class CLIP:
    """
    Wrapper around CLIP to extract image or text features.

    """

    def __init__(self, clip_ver="RN50", device="cpu", debug=False):
        self.clip_model, self.preprocess = clip.load(clip_ver, device=device)
        self.debug = debug
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

    def __init__(self, yolo_ver="yolov5s", clip_ver="RN50", device="cpu", debug=False):
        self.yolo_model = YOLOv5(yolo_ver, debug=debug)
        self.clip_model = CLIP(clip_ver, debug=debug)
        self.debug = debug
        self.device = device

    def __call__(self, img_path, prompt):

        if self.debug:
            print("[INFO] Running YOLO on the image...")

        # Call YOLO to get the bounding boxes of the most relevant objects
        yolo_results = self.yolo_model(img_path)

        if self.debug:
            print(f"[INFO] YOLO found {yolo_results.shape[0]} objects")

        img = Image.open(img_path)

        imgs_encoding = list()

        if self.debug:
            print("[INFO] Running CLIP on detected objects...")

        # Encode each object found
        for i in range(yolo_results.shape[0]):
            bbox = yolo_results.iloc[i, 0:4]

            sub_img = img.crop(bbox)

            imgs_encoding.append(self.clip_model(sub_img))

        imgs_encoding = torch.cat(imgs_encoding, dim=0)

        if self.debug:
            print("[INFO] Running CLIP on the prompt...")

        # Encode the prompt
        text_encoding = self.clip_model(prompt)

        # Compute the distance between the prompt and each object
        dists = (imgs_encoding - text_encoding).norm(dim=-1)

        # Find the best object and the corresponding bounding box
        best_idx = int(dists.argmin())

        ans = yolo_results.iloc[best_idx]

        best_bbox = ans[0:4].tolist()

        rect = patches.Rectangle(
            (best_bbox[0], best_bbox[1]), best_bbox[2] - best_bbox[0], best_bbox[3] - best_bbox[1],
            linewidth=1.5, edgecolor=(0, 1, 0), facecolor='none'
        )

        # Plot the image with the bounding box

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.add_patch(rect)
        ax.axis("off")

        plt.title(f"\"{prompt.capitalize()}\"")
        plt.show()
