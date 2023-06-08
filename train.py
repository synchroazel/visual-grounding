import clip
import torch
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from torchmultimodal.transforms.clip_transform import CLIPTransform, CLIPImageTransform, CLIPTextTransform
from tqdm import tqdm

from modules.refcocog import RefCOCOg
from modules.refcocog import RefCOCOgSample
from modules.utilities import get_best_device

epochs = 10
batch_size = 16
learning_rate = 1e-5

data_path = "dataset/refcocog"  # CHANGE ME

clip_ver = "RN50"

dataset = RefCOCOg(ds_path=data_path, transform_img=CLIPImageTransform(), transform_txt=CLIPTextTransform())

# train_ds = RefCOCOg(ds_path=data_path, split='train')
# val_ds = RefCOCOg(ds_path=data_path, split='val')
# test_ds = RefCOCOg(ds_path=data_path, split='test')

print(f"[INFO] Dataset Size: {len(dataset)}")


# print(f"[INFO] train split:  {len(train_ds)}")
# print(f"[INFO] val split:    {len(val_ds)}")
# print(f"[INFO] test split:   {len(test_ds)}")

# @title Collate function to yield images and sentences from the DataLoader

def collate_fn(batch_):
    batch_ = [RefCOCOgSample(**sample) for sample in batch_]

    images, texts = list(), list()

    for sample in batch_:
        for sentence in sample.sentences:
            images.append(sample.img)
            texts.append(sentence)

    return torch.stack(images), torch.stack(texts)


# @title Main training cell

# get best device
device = get_best_device()

# instantiate clip transform
clip_transform = CLIPTransform()

# instantiate the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# instantiate model. Here we use clip with vit-L as the image encoder
model, _ = clip.load(clip_ver, device=device)

model.float()

# define loss and other things needed for training
contrastive_loss = ContrastiveLossWithTemperature()

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# write your train loop

for n in range(epochs):
    print(f"[INFO] Epoch #{n}")

    pbar = tqdm(dataloader, desc="[INFO] Loss N/A", leave=True)

    for batch in pbar:
        image, text = batch

        image_embeddings, text_embeddings = model(image, text)

        loss = contrastive_loss(image_embeddings.float(), text_embeddings.float())

        if torch.isnan(loss):
            print(f"[INFO] NaN Loss. Skipping...")
            continue
        else:
            pbar.set_description("[INFO] Loss %.4f" % loss)
            loss.backward()
            optimizer.step()

    # save model
    torch.save(model.state_dict(), f"ft_clip-{clip_ver}.pth")
