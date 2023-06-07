import argparse

import clip
import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.customclip import CustomCLIP
from modules.refcocog import RefCOCOgSample, RefCOCOg
from modules.utilities import log_values, get_best_device, get_cost_function, get_optimizer

optimizers = {
    "SGD": torch.optim.SGD,
    "RMSProp": torch.optim.RMSprop,
    "Adam": torch.optim.Adam,
    "Adamax": torch.optim.Adamax,
    "Adadelta": torch.optim.Adadelta,
}

""" Simple fine tuning logic """


def training_step(net, clip_prep_, data_loader, optimizer, cost_function, device):
    n_samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # Set the network to training mode
    net.train()

    # Iterate over the training set
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="[INFO] Training step")):

        inputs, targets = list(), list()

        for sample in batch:
            sample = RefCOCOgSample(**sample)

            prep_img = clip_prep_(sample.img)

            inputs.append(prep_img)
            targets.append(sample.category_id - 1)  # so that category_ids will start from #0

        inputs = torch.stack(inputs)
        targets = torch.tensor(targets)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = net(inputs)

        # Loss computation
        loss = cost_function(outputs, targets)

        loss.backward()

        # Parameters update
        optimizer.step()

        # Gradients reset
        optimizer.zero_grad()

        # Fetch prediction and loss value
        n_samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1)  # max() returns (maximum_value, index_of_maximum_value)

        # Compute training accuracy
        cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / n_samples, cumulative_accuracy / n_samples * 100


def test_step(net, clip_prep_, data_loader, cost_function, device):
    samples_ = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # Set the network to evaluation mode
    net.eval()

    # Disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # Iterate over the test set
        for batch_idx, samples in enumerate(tqdm(data_loader, desc="[INFO] Test step")):

            inputs, targets = list(), list()

            for sample in samples:
                sample = RefCOCOgSample(**sample)

                prep_img = clip_prep_(sample.img)

                inputs.append(prep_img)
                targets.append(sample.category_id - 1)  # so that category_ids will start from #0

            inputs = torch.stack(inputs)
            targets = torch.tensor(targets)

            # Load data into GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = net(inputs)

            # Loss computation
            loss = cost_function(outputs, targets)

            # Fetch prediction and loss value
            samples_ += inputs.shape[0]
            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)

            # Compute accuracy
            cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples_, cumulative_accuracy / samples_ * 100


""" Contrastive Learning training logic """


def contrastive_loss(image_logits, text_logits, cost_function, device):
    labels = np.arange(image_logits.shape[0])
    labels = torch.from_numpy(labels).to(device)

    loss_i = cost_function(image_logits, labels)
    loss_t = cost_function(text_logits, labels)

    return (loss_i + loss_t) / 2.0


def contrastive_training_step(net, clip_prep_, data_loader, optimizer, cost_function, device):
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
                    prep_img = clip_prep_(prep_img)

                    images.append(prep_img)
                    texts.append(sentence)

            texts = clip.tokenize(texts).to(device)
            images = torch.stack(images).to(device)

            images = images.to(device)
            texts = texts.to(device)

            # forward pass
            image_logits, text_logits = net(images, texts)

            # loss computation
            loss = contrastive_loss(image_logits, text_logits, cost_function, device=device)

            # fetch loss value
            n_samples += images.shape[0]
            cumulative_loss += loss.item()

    return cumulative_loss / n_samples, 0



def contrastive_test_step(net, clip_prep_, data_loader, cost_function, device):
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
                    prep_img = clip_prep_(prep_img)

                    images.append(prep_img)
                    texts.append(sentence)

            texts = clip.tokenize(texts).to(device)
            images = torch.stack(images).to(device)

            images = images.to(device)
            texts = texts.to(device)

            # forward pass
            image_logits, text_logits = net(images, texts)

            # loss computation
            loss = contrastive_loss(image_logits, text_logits, cost_function, device=device)

            # fetch loss value
            n_samples += images.shape[0]
            cumulative_loss += loss.item()

    return cumulative_loss / n_samples, 0


""" main (common) training logic """


def training_loop(model,
                  clip_prep_,
                  test_step_fun,
                  training_step_fun,
                  train_ds,
                  val_ds,
                  test_ds,
                  batch_size,
                  learning_rate,
                  weight_decay,
                  momentum,
                  epochs,
                  optimizer,
                  exp_name,
                  runs_dir,
                  device="cpu"):
    # Create a logger for the experiment
    writer = SummaryWriter(log_dir=f"{runs_dir}/{exp_name}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    # Instantiate the network and move it to the chosen device
    net = model.to(device)

    # Instantiate the optimizer
    if exp_name == "clip_contrastive":
        optimizer = get_optimizer(net, learning_rate, weight_decay, momentum, optimizer)
    else:
        optimizer = get_optimizer(net.classifier, learning_rate, weight_decay, momentum, optimizer)

    # Define the cost function
    cost_function = get_cost_function()

    # Computes evaluation results before training
    print('[INFO] Before training:')
    # train_loss, train_accuracy = test_step_fun(net, clip_prep_, train_loader, cost_function, device)
    # val_loss, val_accuracy = test_step_fun(net, clip_prep_, val_loader, cost_function, device)
    test_loss, test_accuracy = test_step_fun(net, clip_prep_, test_loader, cost_function, device)

    # Log to TensorBoard
    # log_values(writer, -1, "train", train_loss, train_accuracy)
    # log_values(writer, -1, "validation", val_loss, val_accuracy)
    log_values(writer, -1, "test", test_loss, test_accuracy)

    # print('\tTraining loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
    # print('\tValidation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
    print('\tTest loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')

    # For each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        print('[INFO] EPOCH: {:d}'.format(e + 1))
        train_loss, train_accuracy = training_step_fun(net, clip_prep_, train_loader, optimizer, cost_function, device)
        val_loss, val_accuracy = test_step_fun(net, clip_prep_, val_loader, cost_function, device)

        # Logs to TensorBoard
        log_values(writer, e, "validation", val_loss, val_accuracy)

        print('\tTraining loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
        print('\tValidation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
        print('-----------------------------------------------------')

    # Compute final evaluation results
    print('[INFO] After training:')
    train_loss, train_accuracy = test_step_fun(net, clip_prep_, train_loader, cost_function, device)
    val_loss, val_accuracy = test_step_fun(net, clip_prep_, val_loader, cost_function, device)
    test_loss, test_accuracy = test_step_fun(net, clip_prep_, test_loader, cost_function, device)

    # Log to TensorBoard
    log_values(writer, epochs, "train", train_loss, train_accuracy)
    log_values(writer, epochs, "validation", val_loss, val_accuracy)
    log_values(writer, epochs, "test", test_loss, test_accuracy)

    print('\t[INFO] Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
    print('\t[INFO] Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
    print('\t[INFO] Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')

    # Closes the logger
    writer.close()

    # Save model checkpoint to disk
    torch.save(net.state_dict(), f"{runs_dir}/{exp_name}/{exp_name}.pth")


def main(args):
    if not (args.contrastive ^ args.simple):
        raise ValueError(f"Please choose either --contrastive or --simple (classifier) training.")

    device = get_best_device()

    dataset = RefCOCOg(ds_path=args.datapath)
    train_ds = RefCOCOg(ds_path=args.datapath, split='train')
    val_ds = RefCOCOg(ds_path=args.datapath, split='val')
    test_ds = RefCOCOg(ds_path=args.datapath, split='test')

    if args.red_dataset is not None:
        print(f"[INFO] Reducing dataset to {args.red_dataset * 100}% of its original size.")
        keep = args.red_dataset
        dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
        train_ds, _ = random_split(train_ds, [int(keep * len(train_ds)), len(train_ds) - int(keep * len(train_ds))])
        val_ds, _ = random_split(val_ds, [int(keep * len(val_ds)), len(val_ds) - int(keep * len(val_ds))])
        test_ds, _ = random_split(test_ds, [int(keep * len(test_ds)), len(test_ds) - int(keep * len(test_ds))])

    print(f"[INFO] Dataset Size: {len(dataset)}")
    print(f"[INFO] train split:  {len(train_ds)}")
    print(f"[INFO] val split:    {len(val_ds)}")
    print(f"[INFO] test split:   {len(test_ds)}")

    clip_model, clip_prep = clip.load("RN50", device=device)

    print("[INFO] Hyperparameters:")
    print("[INFO]   batch size = ", args.batch_size)
    print("[INFO]   epochs = ", args.epochs)
    print("[INFO]   learning rate = ", args.learning_rate)
    print("[INFO]   weight decay = ", args.weight_decay)
    print("[INFO]   momentum = ", args.momentum)
    print("[INFO]   optimizer = ", args.optimizer)

    if args.contrastive:
        print("[INFO] CLIP will be fine-tuned using Contrastive Learning on all RefCOCOg samples.")

        training_loop(model=clip_model,
                      clip_prep_=clip_prep,
                      test_step_fun=contrastive_test_step,
                      training_step_fun=contrastive_training_step,
                      train_ds=train_ds,
                      val_ds=val_ds,
                      test_ds=test_ds,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      weight_decay=args.weight_decay,
                      momentum=args.momentum,
                      optimizer=args.optimizer,
                      runs_dir=args.runs_dir,
                      exp_name="clip_contrastive",
                      device=device)

    elif args.simple:
        print("[INFO] CLIP will be fine-tuned using a simple classifier on RefCOCOg classes.")
        custom_clip_model = CustomCLIP(num_classes=90).to(device)  # 90 classes in RefCOCOg

        training_loop(model=custom_clip_model,
                      clip_prep_=clip_prep,
                      test_step_fun=test_step,
                      training_step_fun=training_step,
                      train_ds=train_ds,
                      val_ds=val_ds,
                      test_ds=test_ds,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      weight_decay=args.weight_decay,
                      momentum=args.momentum,
                      optimizer=args.optimizer,
                      runs_dir=args.runs_dir,
                      exp_name="clip_classifier",
                      device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine tune CLIP on RefCOCOg')

    parser.add_argument('-c', '--contrastive', action='store_true',
                        help='Fine-tune CLIP using Contrastive Learning on RefCOCOg')
    parser.add_argument('-s', '--simple', action='store_true',
                        help='Fine-tune CLIP on RefCOCOg using a simple classifier')
    parser.add_argument('-dp', '--datapath', type=str, default="dataset/refcocog",
                        help='path to the dataset.')
    parser.add_argument('-rd', '--red_dataset', type=float, default=0.1,
                        help='Whether to use a reduced version of the dataset or not')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to train the model for')
    parser.add_argument('-bs', '--batch_size', type=int, default=128,
                        help='Batch size to use during training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate to use during training')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='Momentum to use during training')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6,
                        help='Weight decay to use during training')
    parser.add_argument('-o', '--optimizer', type=str, default="Adam",
                        help='Optimizer to use during training')
    parser.add_argument('-r', '--runs_dir', type=str, default="runs",
                        help='Directory where to save the runs')

    args = parser.parse_args()

    # todo: remove
    args.contrastive = True
    args.datapath = "/media/dmmp/vid+backup/Data/refcocog"
    args.batch_size = 128

    main(args)
