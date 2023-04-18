import matplotlib.pyplot as plt


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
