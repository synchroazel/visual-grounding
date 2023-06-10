import argparse

import clip
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from torchmultimodal.transforms.clip_transform import CLIPImageTransform, CLIPTextTransform
from tqdm import tqdm

from modules.refcocog import RefCOCOg
from modules.refcocog import RefCOCOgSample
from modules.utilities import get_best_device


def main(args):
    # Get the best device for the current machine
    device = get_best_device()

    # Load the (full) dataset
    dataset = RefCOCOg(ds_path=args.datapath, split="train",
                       transform_img=CLIPImageTransform(),
                       transform_txt=CLIPTextTransform())

    # The `collate_fn` function handles the creation of batched in the dataloader
    def collate_fn(batch_):
        batch_ = [RefCOCOgSample(**sample) for sample in batch_]

        images, texts = list(), list()

        for sample in batch_:
            for sentence in sample.sentences:
                images.append(sample.img)
                texts.append(sentence)

        return torch.stack(images), torch.stack(texts)

    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    # Instantiate the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Instantiate CLIP model
    clip_model, _ = clip.load(args.clip_version, device=device, jit=False)
    del _

    # Save a name for the model checkpoint
    model_pt_name = f"ft-clip-{args.clip_version.replace('/', '-')}.pt"

    # Load the model if want to resume training
    if args.resume:
        checkpoint = torch.load(model_pt_name)
        clip_model.load_state_dict(checkpoint['model_state_dict'])
        last_epoch = checkpoint['epoch']
        del checkpoint
        print(f"[INFO] Loaded model from {model_pt_name}")
    resumed = False

    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {trainable_params}")

    # Freeze all layers except the last two
    for param in clip_model.parameters():
        param.requires_grad = False
    for param in clip_model.transformer.parameters():
        param.requires_grad = True

    # Check how many layers are trainable
    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {trainable_params}")

    # Set model precision according to device
    if device == torch.device("cpu") or torch.device("mps"):
        clip_model.float()
    else:
        clip.model.convert_weights(clip_model)

    # Define loss function
    contrastive_loss = ContrastiveLossWithTemperature()

    # Define optimizer
    # optimizer = torch.optim.AdamW(clip_model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    # Create a logger for the experiment
    writer = SummaryWriter(log_dir=f"{args.runs_dir}/clip-ft-{args.clip_version}")

    epoch_losses = list()

    print(f"[INFO] Model precision: {clip_model.dtype}")

    for epoch in range(args.epochs):

        if args.resume and not resumed:
            if epoch <= last_epoch:
                continue
            else:
                print(f"\n[INFO] Resuming from epoch #{epoch}")
                resumed = True

        print(f"\n[INFO] Epoch #{epoch}")

        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc="[INFO] Loss N/A", leave=True)

        for batch in pbar:
            image, text = batch

            image_embeddings, text_embeddings = clip_model(image.to(device), text.to(device))

            loss = contrastive_loss(image_embeddings, text_embeddings)

            epoch_losses.append(loss)

            if torch.isnan(loss):
                print(f"\n[WARN] NaN Loss! Skipping...\n")
                continue

            pbar.set_description("[INFO] Loss %.4f" % loss)

            loss.backward()

            if device == torch.device("cpu") or torch.device("mps"):
                if args.weight_clipping:
                    torch.nn.utils.clip_grad_value_(clip_model.parameters(), clip_value=100.0)
                optimizer.step()
            else:
                convert_models_to_fp32(clip_model)
                if args.weight_clipping:
                    torch.nn.utils.clip_grad_value_(clip_model.parameters(), clip_value=100.0)
                optimizer.step()
                clip.model.convert_weights(clip_model)

        # Log to tensorboard
        writer.add_scalar(f"loss", torch.mean(torch.tensor(epoch_losses)), epoch)

        print(f"[INFO] Avg. epoch loss: {torch.mean(torch.tensor(epoch_losses))}")

        # Save the model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': clip_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': torch.mean(torch.tensor(epoch_losses)),
        }, model_pt_name)

    # Closes the logger
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine tune CLIP on RefCOCOg')

    parser.add_argument('-dp', '--datapath', type=str, default="dataset/refcocog",
                        help='path to the dataset.')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to train the model for')
    parser.add_argument('-bs', '--batch_size', type=int, default=16,
                        help='Batch size to use during training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5,
                        help='Learning rate to use during training')
    parser.add_argument('-cv', '--clip_version', type=str, default="RN50",
                        help='CLIP version to use (RN50, RN101, ViT-L/14)')
    parser.add_argument('-rd', '--runs_dir', type=str, default="runs",
                        help='Directory where to save the runs')
    parser.add_argument('-f64', '--f64', action='store_true',
                        help='Use float64 parameters')
    parser.add_argument('-wc', '--weight_clipping', action='store_true',
                        help='Use weight clipping')
    parser.add_argument('-rs', '--resume', action='store_true',
                        help='Resume training from last checkpoint')

    args = parser.parse_args()

    main(args)
