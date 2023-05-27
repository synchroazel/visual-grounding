from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from modules.refcocog import RefCOCOgSample

OPTIMIZERS_TO_TRY = {
    "SGD": torch.optim.SGD,
    "RMSProp": torch.optim.RMSprop,
    "Adam": torch.optim.Adam,
    "Adamax": torch.optim.Adamax,
    "Adadelta": torch.optim.Adadelta,
    # todo: add more
}


# tensorboard logging utilities
def log_values(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
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


def quality_f(matrix):
    # less is more :)
    return matrix.sum()


def find_best_bbox(heatmap, lower_bound=-1.0, upper_bound=1.0):
    # Rescale the heatmap
    heatmap = MinMaxScaler(feature_range=(lower_bound, upper_bound)).fit_transform(heatmap)

    # Initialize the best score and best box
    best_score = float('-inf')
    best_box = None

    # Loop over all possible box sizes and positions
    for w in range(1, heatmap.shape[1] + 1):
        for h in range(1, heatmap.shape[0] + 1):
            for i in range(heatmap.shape[1] - w + 1):
                for j in range(heatmap.shape[0] - h + 1):

                    # Get current sub-region
                    candidate = heatmap[j:j + h, i:i + w]

                    # Compute the score for this box
                    score = quality_f(candidate)

                    # Update the best score and best box if necessary
                    if score > best_score:
                        best_score = score
                        best_box = (i, j, w, h)

    return best_box


def visual_grounding_test(vg_pipeline, dataset, logging=False):
    scores = list()

    pbar = tqdm(dataset)

    for sample in pbar:

        sample = RefCOCOgSample(**sample)

        for sentence in sample.sentences:

            sc = vg_pipeline(sample, sentence, show=False)

            scores.append(sc)

            avg_metrics = list()

            # The bar description is live updated with the average score for each metric

            for metric in scores[0].keys():
                avg_metric = np.mean([score[metric] for score in scores if score[metric] is not None])
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
        avg_metric = np.mean([score[metric] for score in scores])

        print("Avg. {}: {:.3f}".format(metric, avg_metric))
