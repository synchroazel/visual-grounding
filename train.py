import argparse
from datetime import datetime
import clip
import torch
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from torchmultimodal.transforms.clip_transform import CLIPImageTransform, CLIPTextTransform
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from modules.refcocog import RefCOCOg
from modules.refcocog import RefCOCOgSample
from modules.utilities import get_best_device


def main(args):
    # Get the best device for the current machine
    device = get_best_device()

    # Load the (full) dataset
    dataset = RefCOCOg(ds_path=args.datapath, transform_img=CLIPImageTransform(), transform_txt=CLIPTextTransform())

    # The `collate_fn` function handles the creation of batched in the dataloader
    def collate_fn(batch_):
        batch_ = [RefCOCOgSample(**sample) for sample in batch_]

        images, texts = list(), list()

        for sample in batch_:
            for sentence in sample.sentences:
                images.append(sample.img)
                texts.append(sentence)

        return torch.stack(images), torch.stack(texts)

    # Instantiate the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Instantiate CLIP model
    clip_model, _ = clip.load(args.clip_version, device=device)

    # Cast the model to float
    clip_model.float()

    # Define loss function
    contrastive_loss = ContrastiveLossWithTemperature()

    # Define optimizer
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=args.learning_rate)

    # Create a logger for the experiment
    datetag = datetime.now().strftime("%m%d%H%M%S")
    writer = SummaryWriter(log_dir=f"{args.runs_dir}/clip-ft-{args.clip_version}-{datetag}")

    def log_values(writer, step, prefix, loss):
        writer.add_scalar(f"{prefix}/loss", loss, step)

    for n in range(args.epochs):
        print(f"\n[INFO] Epoch #{n}")

        pbar = tqdm(dataloader, desc="[INFO] Loss N/A", leave=True)

        for batch in pbar:
            image, text = batch

            image_embeddings, text_embeddings = clip_model(image.to(device), text.to(device))

            loss = contrastive_loss(image_embeddings, text_embeddings)

            # Log to tensorboard
            writer.add_scalar(f"loss", loss, n)

            if torch.isnan(loss):
                print(f"\n[WARN] NaN Loss! Skipping...\n")
                continue
            else:
                pbar.set_description("[INFO] Loss %.4f" % loss)
                loss.backward()
                optimizer.step()

        # Closes the logger
        writer.close()

        # Save the model after each epoch
        model_name = f"ft_clip-{args.clip_version.replace('/', '-')}.pth"
        torch.save(clip_model.state_dict(), model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine tune CLIP on RefCOCOg')

    parser.add_argument('-dp', '--datapath', type=str, default="dataset/refcocog",
                        help='path to the dataset.')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to train the model for')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size to use during training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5,
                        help='Learning rate to use during training')
    parser.add_argument('-cv', '--clip_version', type=str, default="RN50",
                        help='CLIP version to use (RN50, RN101, ViT-L/14)')
    parser.add_argument('-r', '--runs_dir', type=str, default="runs",
                        help='Directory where to save the runs')

    args = parser.parse_args()

    main(args)
