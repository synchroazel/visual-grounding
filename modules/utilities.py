import matplotlib.pyplot as plt
import torch

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
            bbox={"facecolor": (0, 1, 0), "pad": 2, "color": (0, 1, 0)})

    ax.add_patch(gt_rect)
    ax.text(gt_bbox[0], gt_bbox[3], "true", color=(1, 1, 1),
            bbox={"facecolor": (1, 0, 0), "pad": 2, "color": (1, 0, 0)})

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
