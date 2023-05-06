import numpy as np
import torch

OPTIMIZERS_TO_TRY = {
    "SGD":torch.optim.SGD,
    "RMSProp":torch.optim.RMSprop,
    "Adam":torch.optim.Adam,
    "Adamax":torch.optim.Adamax,
    "Adadelta":torch.optim.Adadelta,
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


