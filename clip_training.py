import json
import os.path
import pickle

import clip
import numpy as np
from PIL import Image
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.clip_classes import CustomCLIP
from modules.refcocog import RefCOCOg, RefCOCOgSample
from modules.utilities import *

if torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA GPU
    print("[INFO] Using GPU.")
elif torch.has_mps:
    device = torch.device("mps")  # Apple Silicon GPU
    print("[INFO] Using MPS.")
else:
    device = torch.device("cpu")
    print("[INFO] No GPU found, using CPU instead.")

# HYPERPARAMETERS

batch_size = 64  # 128  # 256 causes out of memory with 24GB of GPU ram
learning_rate = 0.001
momentum = 0.9
epochs = 3
optimizer = "Adam"

clip_model, clip_prep = clip.load("RN50", device=device, jit=False)

print("[INFO] Model params: {:,}".format(np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()])))
print("[INFO] Trainable params: {:,}".format(sum(p.numel() for p in clip_model.parameters() if p.requires_grad)))
print("[INFO] Input resolution: ", clip_model.visual.input_resolution)
print("[INFO] Max prompt length:", clip_model.context_length)
print("[INFO] Vocab size:", clip_model.vocab_size)

dataset_path = "/media/dmmp/vid+backup/Data/refcocog"
# data_path = "../dataset/refcocog"

dataset = RefCOCOg(ds_path=dataset_path)

train_ds = RefCOCOg(ds_path=dataset_path, split='train')
val_ds = RefCOCOg(ds_path=dataset_path, split='val')
test_ds = RefCOCOg(ds_path=dataset_path, split='test')

print("Keeping toy dataset")
keep = 0.1
dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
train_ds, _ = random_split(train_ds, [int(keep * len(train_ds)), len(train_ds) - int(keep * len(train_ds))])
val_ds, _ = random_split(val_ds, [int(keep * len(val_ds)), len(val_ds) - int(keep * len(val_ds))])
test_ds, _ = random_split(test_ds, [int(keep * len(test_ds)), len(test_ds) - int(keep * len(test_ds))])

print(f"Dataset Size: {len(dataset)}\n")
print(f"Train size: {len(train_ds)}")
print(f"Val size:   {len(val_ds)}")
print(f"Test size:  {len(test_ds)}")


# ----------------------------------------------------------------------------------------------------------------------
def get_data(dataset):
    texts, images = list(), list()

    for sample in tqdm(dataset, desc="[INFO] Loading images and captions"):
        sample = RefCOCOgSample(**sample)

        for sentence in sample.sentences:
            images.append(sample.path)
            texts.append(sentence)

    return images, texts


def encode_data(images_fp: list[str], texts: list[str]):
    global device
    # preprocess the images to transform from filenames to images to tensors
    images = [clip_prep(Image.open(image)) for image in tqdm(images_fp, desc="[INFO] Preprocessing images")]
    images = torch.tensor(np.stack(images)).to(device)

    # preprocess the texts to transform from text to tensors
    text_tokens = clip.tokenize(["This is " + desc for desc in tqdm(texts, desc="[INFO] Preprocessing texts")]).to(
        device)

    # encode the inputs
    with torch.no_grad():
        print("[INFO] Encoding images...")
        images_z = clip_model.encode_image(images).float()
        print("[INFO] Encoding texts...")
        texts_z = clip_model.encode_text(text_tokens).float()

    return images_z, texts_z


def training_step_ft(net, data_loader, optimizer, cost_function, device=device):
    n_samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # set the network to training mode
    net.train()

    # iterate over the training set
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="[INFO] Training step")):

        inputs, targets = list(), list()

        for sample in batch:
            sample = RefCOCOgSample(**sample)

            prep_img = clip_prep(sample.img)

            inputs.append(prep_img)
            targets.append(sample.category_id - 1)  # so that category_ids will start from #0

        inputs = torch.stack(inputs)
        targets = torch.tensor(targets)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward pass
        outputs = net(inputs)

        # loss computation
        loss = cost_function(outputs, targets)

        # backward pass
        loss.backward()

        # parameters update
        optimizer.step()

        # gradients reset
        optimizer.zero_grad()

        # fetch prediction and loss value
        n_samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1)  # max() returns (maximum_value, index_of_maximum_value)

        # compute training accuracy
        cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / n_samples, cumulative_accuracy / n_samples * 100


def test_step_ft(net, data_loader, cost_function, device=device):
    samples_ = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # set the network to evaluation mode
    net.eval()

    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for batch_idx, samples in enumerate(tqdm(data_loader, desc="[INFO] Test step")):

            inputs, targets = list(), list()

            for sample in samples:
                sample = RefCOCOgSample(**sample)

                prep_img = clip_prep(sample.img)

                inputs.append(prep_img)
                targets.append(sample.category_id - 1)  # so that category_ids will start from #0

            inputs = torch.stack(inputs)
            targets = torch.tensor(targets)

            # load data into GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward pass
            outputs = net(inputs)

            # loss computation
            loss = cost_function(outputs, targets)

            # fetch prediction and loss value
            samples_ += inputs.shape[0]
            cumulative_loss += loss.item()  # Note: the .item() is needed to extract scalars from tensors
            _, predicted = outputs.max(1)

            # compute accuracy
            cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples_, cumulative_accuracy / samples_ * 100


def training_loop_ft(train_ds,
                     val_ds,
                     test_ds,
                     batch_size=batch_size,
                     num_classes=90,  # 90 classes in RefCOCOg
                     device=device,
                     learning_rate=learning_rate,
                     weight_decay=0.000001,
                     momentum=momentum,
                     epochs=epochs,
                     optimizer=optimizer):
    # create a logger for the experiment
    writer = SummaryWriter(log_dir="runs/exp1")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    # instantiate the network and move it to the chosen device (GPU)
    net = CustomCLIP(num_classes=num_classes).to(device)

    # instantiate the optimizer
    optimizer = get_optimizer_ft(net, learning_rate, weight_decay, momentum, optimizer)

    # define the cost function
    cost_function = get_cost_function()

    # computes evaluation results before training
    print('Before training:')

    test_loss, test_accuracy = test_step_ft(net, test_loader, cost_function)

    # log to TensorBoard

    log_values(writer, -1, test_loss, test_accuracy, "test")

    print('\tTest loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')

    # for each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        train_loss, train_accuracy = training_step_ft(net, train_loader, optimizer, cost_function)
        val_loss, val_accuracy = test_step_ft(net, val_loader, cost_function)

        # logs to TensorBoard
        log_values(writer, e, val_loss, val_accuracy, "Validation")

        print('Epoch: {:d}'.format(e + 1))
        print('\tTraining loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
        print('\tValidation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
        print('-----------------------------------------------------')

    # compute final evaluation results
    print('After training:')

    test_loss, test_accuracy = test_step_ft(net, test_loader, cost_function)

    # log to TensorBoard

    log_values(writer, epochs, test_loss, test_accuracy, "test")

    print('\tTest loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')

    # closes the logger
    writer.close()
    return net


# ----------------------------------------------------------------------------------------------------------------------
def contrastive_loss(image_logits, text_logits, cost_function):
    labels = np.arange(image_logits.shape[0])
    labels = torch.from_numpy(labels).to(device)

    loss_i = cost_function(image_logits, labels)
    loss_t = cost_function(text_logits, labels)

    return (loss_i + loss_t) / 2.0


def training_step_cl(net, data_loader, optimizer, cost_function, device=device):
    n_samples = 0.0
    cumulative_loss = 0.0

    # set the network to training mode
    net.train()

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="[INFO] Training step")):

        images, texts = list(), list()

        for sample in batch:
            sample = RefCOCOgSample(**sample)

            for sentence in sample.sentences:
                prep_img = sample.img.crop(sample.bbox)
                prep_img = clip_prep(prep_img)

                images.append(prep_img)
                texts.append(sentence)

        texts = clip.tokenize(texts).to(device)
        images = torch.stack(images).to(device)

        # forward pass
        image_logits, text_logits = net(images, texts)

        # loss computation
        loss = contrastive_loss(image_logits, text_logits, cost_function)

        # backward pass
        loss.backward()

        # parameters update
        optimizer.step()

        # gradients reset
        optimizer.zero_grad()

        # fetch loss value
        n_samples += images.shape[0]
        cumulative_loss += loss.item()

    return cumulative_loss / n_samples


def test_step_cl(net, data_loader, cost_function, device=device):
    n_samples = 0.0
    cumulative_loss = 0.0

    # set the network to evaluation mode
    net.eval()

    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm(data_loader, desc="[INFO] Test step")):

            images, texts = list(), list()

            for sample in batch:
                sample = RefCOCOgSample(**sample)

                for sentence in sample.sentences:
                    prep_img = sample.img.crop(sample.bbox)
                    prep_img = clip_prep(prep_img)

                    images.append(prep_img)
                    texts.append(sentence)

            texts = clip.tokenize(texts).to(device)
            images = torch.stack(images).to(device)

            images = images.to(device)
            texts = texts.to(device)

            # forward pass
            image_logits, text_logits = net(images, texts)

            # image logits are all nan
            # loss computation
            loss = contrastive_loss(image_logits, text_logits, cost_function)

            # fetch loss value
            n_samples += images.shape[0]
            cumulative_loss += loss.item()

    return cumulative_loss / n_samples


def training_loop_cl(train_ds,
                     val_ds,
                     test_ds,
                     batch_size=batch_size,
                     num_classes=90,  # 90 classes in RefCOCOg
                     device=device,
                     learning_rate=learning_rate,
                     weight_decay=0.000001,
                     momentum=momentum,
                     epochs=epochs,
                     optimizer=optimizer):
    # create a logger for the experiment
    global clip_model
    writer = SummaryWriter(log_dir="runs/exp1")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    # instantiate the network and move it to the chosen device (GPU)
    net = clip_model.to(device)
    net = net.float()

    # instantiate the optimizer
    optimizer = get_optimizer_cl(net, learning_rate, weight_decay, momentum, optimizer)

    # define the cost function
    cost_function = get_cost_function()

    # computes evaluation results before training
    # tODO: fix error here: CustomCLIP.forward() takes 2 positional arguments but 3 were given
    print('Before training:')
    test_loss = test_step_cl(net, test_loader, cost_function)

    # print(train_loss)

    # log to TensorBoard error for coffe2
    # log_values(writer, -1, test_loss,accuracy="Not applicable", prefix="test")

    print('\tTest loss {:.5f}'.format(test_loss))
    print('-----------------------------------------------------')

    # for each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        train_loss = training_step_cl(net, train_loader, optimizer, cost_function)
        val_loss = test_step_cl(net, val_loader, cost_function)

        # logs to TensorBoard
        # log_values(writer, e, val_loss, "Validation")

        print('Epoch: {:d}'.format(e + 1))
        print('\tTraining loss {:.5f}'.format(train_loss))
        print('\tValidation loss {:.5f}'.format(val_loss))
        print('-----------------------------------------------------')

    # compute final evaluation results
    print('After training:')
    train_loss = test_step_cl(net, train_loader, cost_function)
    val_loss = test_step_cl(net, val_loader, cost_function)
    test_loss = test_step_cl(net, test_loader, cost_function)

    # log to TensorBoard
    # log_values(writer, epochs, train_loss, "train")
    # log_values(writer, epochs, val_loss, "validation")
    # log_values(writer, epochs, test_loss, "test")

    print('\tTraining loss {:.5f}'.format(train_loss))
    print('\tValidation loss {:.5f}'.format(val_loss))
    print('\tTest loss {:.5f}'.format(test_loss))
    print('-----------------------------------------------------')

    # closes the logger
    writer.close()
    return net


# ----------------------------------------------------------------------------------------------------------------------

def visual_grounding_test(vg_pipeline, dataset):
    scores = list()

    for sample in tqdm(dataset, desc=f"Testing on {len(dataset)} images"):

        sample = RefCOCOgSample(**sample)

        for sentence in sample.sentences:

            try:
                sc = vg_pipeline(sample, sentence, show=False)
            except ValueError:
                continue

            scores.append(sc)

    for metric in scores[0].keys():
        avg_metric = np.mean([score[metric] for score in scores])

        print("Avg. {}: {:.3f}".format(metric, avg_metric))
    return scores


cases = ("fine_tune_clip", "contrastive_learning", "test_base_clip", "test_ft_clip", "test_contrastive_clip")

case = "contrastive_learning"
match case:
    case "fine_tune_clip":  # fine-tuned clip

        save_path = "saved_models"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fp = os.path.join(save_path, "fine_tuned_clip.pickle")
        fine_tuned_clip = training_loop_ft(train_ds, val_ds, test_ds)
        with open(fp, 'wb') as f:
            pickle.dump(fine_tuned_clip, f)
        print("Model saved as: " + fp)

    case "contrastive_learning":  # contrastive learning clip

        save_path = "saved_models"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        batch_size = 64
        fp = os.path.join(save_path, "contrastive_loss_clip.pickle")
        contrastive_loss_clip = training_loop_cl(train_ds, val_ds, test_ds)
        with open(fp, 'wb') as f:
            pickle.dump(contrastive_loss_clip, f)
        print("Model saved as: " + fp)

    case "test_base_clip":

        save_path = "results"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        from modules.yoloclip import YoloClip

        yoloclip = YoloClip(device=device, categories=dataset.dataset.categories)

        scores = visual_grounding_test(yoloclip, test_ds)

        fp = os.path.join(save_path, 'results_base_clip_yolov8x.json')
        with open(fp, 'w') as f:
            json.dump(scores, f)
        print("results saved as: " + fp)

    case "test_ft_clip":
        load_path = "saved_models"
        save_path = "results"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fp = os.path.join(load_path, "fine_tuned_clip.pickle")
        with open(fp, 'rb') as f:
            clip_model = pickle.load(f)

        from modules.yoloclip import YoloClip

        yoloclip = YoloClip(device=device, categories=dataset.dataset.categories)
        yoloclip.clip_model = clip_model
        scores = visual_grounding_test(yoloclip, test_ds)

        fp = os.path.join(save_path, 'results_fine_clip_yolov8x.json')
        with open(fp, 'w') as f:
            json.dump(scores, f)
        print("results saved as: " + fp)

    case "test_contrastive_clip":
        load_path = "saved_models"
        save_path = "results"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fp = os.path.join(load_path, "contrastive_loss_clip.pickle")
        with open(fp, 'rb') as f:
            clip_model = pickle.load(f)

        from modules.yoloclip import YoloClip

        yoloclip = YoloClip(device=device, categories=dataset.dataset.categories)
        yoloclip.clip_model = clip_model
        scores = visual_grounding_test(yoloclip, test_ds)

        fp = os.path.join(save_path, 'results_contrastive_clip_yolov8x.json')
        with open(fp, 'w') as f:
            json.dump(scores, f)
        print("results saved as: " + fp)
