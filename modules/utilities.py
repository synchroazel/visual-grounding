from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from modules.refcocog import RefCOCOgSample

OPTIMIZERS_TO_TRY = {
    "SGD": torch.optim.SGD,
    "RMSProp": torch.optim.RMSprop,
    "Adam": torch.optim.Adam,
    "Adamax": torch.optim.Adamax,
    "Adadelta": torch.optim.Adadelta,
    # TODO: add more
}


def IoU(true_bbox, predicted_bbox):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(true_bbox[0], predicted_bbox[0])
    yA = max(true_bbox[1], predicted_bbox[1])
    xB = min(true_bbox[2], predicted_bbox[2])
    yB = min(true_bbox[3], predicted_bbox[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    true_bboxArea = (true_bbox[2] - true_bbox[0] + 1) * (true_bbox[3] - true_bbox[1] + 1)
    predicted_bboxArea = (predicted_bbox[2] - predicted_bbox[0] + 1) * (predicted_bbox[3] - predicted_bbox[1] + 1)

    # Compute the intersection over union by taking the intersection area
    # and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / float(true_bboxArea + predicted_bboxArea - interArea)

    return iou


def get_best_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # CUDA GPU
        print("[INFO] Using cuda.")
    elif torch.has_mps:
        device = torch.device("mps")  # Apple Silicon GPU
        print("[INFO] Using MPS.")
    else:
        device = torch.device("cpu")
        print("[INFO] No GPU found, using CPU instead.")

    return device


def get_data(dataset):
    texts, images = list(), list()

    for sample in tqdm(dataset, desc="[INFO] Loading images and captions"):
        sample = RefCOCOgSample(**sample)

        for sentence in sample.sentences:
            images.append(sample.path)
            texts.append(sentence)

    return images, texts


def get_optimizer(layer, lr, wd, momentum, optimizer):
    try:
        optimizer = OPTIMIZERS_TO_TRY[optimizer]([
            {'params': layer.parameters(), 'lr': lr}
        ], lr=lr, weight_decay=wd, momentum=momentum)
    except TypeError:
        optimizer = OPTIMIZERS_TO_TRY[optimizer]([
            {'params': layer.parameters(), 'lr': lr}
        ], lr=lr, weight_decay=wd)

    return optimizer


def log_values(writer, step, prefix, loss, accuracy=None):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    if accuracy is not None:
        writer.add_scalar(f"{prefix}/accuracy", accuracy, step)


def cosine_similarity(images_z: torch.Tensor, texts_z: torch.Tensor):
    # normalise the image and the text
    images_z /= images_z.norm(dim=-1, keepdim=True)
    texts_z /= texts_z.norm(dim=-1, keepdim=True)

    # evaluate the cosine similarity between the sets of features
    similarity = (texts_z @ images_z.T)

    return similarity


def display_preds(img, prompt, pred_bbox, gt_bbox, model_name):
    fig, ax = plt.subplots()
    ax.imshow(img)

    pred_rect = plt.Rectangle(
        (pred_bbox[0], pred_bbox[1]), pred_bbox[2] - pred_bbox[0], pred_bbox[3] - pred_bbox[1],
        linewidth=1.5, edgecolor=(0, 1, 0), facecolor='none'
    )

    gt_rect = plt.Rectangle(
        (gt_bbox[0], gt_bbox[1]), gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1],
        linewidth=1.5, edgecolor=(1, 0, 0), facecolor='none'
    )

    ax.add_patch(pred_rect)
    ax.text(pred_bbox[0], pred_bbox[1], "predicted", color=(1, 1, 1),
            bbox={"facecolor": (0, 1, 0), "edgecolor": (0, 1, 0), "pad": 2})

    ax.add_patch(gt_rect)
    ax.text(gt_bbox[0], gt_bbox[3], "true", color=(1, 1, 1),
            bbox={"facecolor": (1, 0, 0), "edgecolor": (1, 0, 0), "pad": 2})

    ax.axis("off")
    plt.title(f"\"{prompt.capitalize()}\"\n")
    plt.text(0.5, -0.075, f"using {model_name}", size=10, ha="center", transform=ax.transAxes)
    plt.show()


def get_optimizer_ft(model, lr, wd, momentum, optimizer):
    try:
        optimizer = OPTIMIZERS_TO_TRY[optimizer]([
            {'params': model.classifier.parameters(), 'lr': lr}
        ], lr=lr, weight_decay=wd, momentum=momentum)
    except TypeError:
        optimizer = OPTIMIZERS_TO_TRY[optimizer]([
            {'params': model.classifier.parameters(), 'lr': lr}
        ], lr=lr, weight_decay=wd)

    return optimizer


def get_optimizer_cl(model, lr, wd, momentum, optimizer):
    try:
        optimizer = OPTIMIZERS_TO_TRY[optimizer]([
            {'params': model.visual.layer4.parameters(), 'lr': lr}
        ], lr=lr, weight_decay=wd, momentum=momentum)
    except TypeError:
        optimizer = OPTIMIZERS_TO_TRY[optimizer]([
            {'params': model.visual.layer4.parameters(), 'lr': lr}
        ], lr=lr, weight_decay=wd)

    return optimizer


def get_cost_function():
    cost_function = torch.nn.CrossEntropyLoss()
    return cost_function


def downsample_map(map, factor):
    # number of blocks in each dimension
    blocks_h = map.shape[0] // factor
    blocks_w = map.shape[1] // factor

    # reshape the original matrix into blocks
    blocks = map[:blocks_h * factor, :blocks_w * factor].reshape(blocks_h, factor, blocks_w, factor)

    # calculate the average of each block
    averages = blocks.mean(axis=(1, 3))

    return averages


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    Args:
        bbox (~numpy.ndarray): See the table below.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def visual_grounding_test(vg_pipeline, dataset, logging=False):
    scores = list()

    pbar = tqdm(dataset)

    for sample in pbar:

        sample = RefCOCOgSample(**sample)

        for sentence in sample.sentences:

            sc = vg_pipeline(sample, sentence, show=False)

            scores.append(sc)

            avg_metrics = list()

            for metric in scores[0].keys():
                avg_metric = np.mean([score[metric] for score in scores if score[metric] is not np.nan])
                avg_metric = f"{metric}: {avg_metric:.3f}"
                avg_metrics.append(avg_metric)

            pbar_desc = " | ".join(avg_metrics)

            if logging:
                pipeline_name = vg_pipeline.__class__.__name__.lower()
                datetime_tag = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                with open(f"logs/{pipeline_name}_log_{datetime_tag}.txt", "a") as f:
                    f.write("[" + datetime_tag + "] " + pbar_desc + "\n")

            pbar.set_description(pbar_desc)

    for metric in scores[0].keys():
        avg_metric = np.mean([score[metric] for score in scores if score[metric] is not np.nan])

        print("Avg. {}: {:.3f}".format(metric, avg_metric))

    return scores
