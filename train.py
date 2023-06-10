import clip
import torch
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from torchmultimodal.transforms.clip_transform import CLIPTransform, CLIPImageTransform, CLIPTextTransform
from tqdm import tqdm

from modules.refcocog import RefCOCOg
from modules.refcocog import RefCOCOgSample
from modules.utilities import get_best_device

torch.autograd.set_detect_anomaly(True)

epochs = 10
batch_size = 32

data_path = "/media/dmmp/vid+backup/Data/refcocog"  # CHANGE ME

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
# model, _ = clip.load("ViT-L/14", device=device)
model, _ = clip.load("RN101", device=device)

model = model.float()
del _
# define loss and other things needed for training
contrastive_loss = ContrastiveLossWithTemperature()

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# write your train loop

for n in range(epochs):
    print(f"[INFO] Epoch #{n}")

    pbar = tqdm(dataloader, desc="[INFO] Loss N/A", leave=True)

    for batch in pbar:
        image, text = batch
        # image, text = clip_transform(image, text)

        image_embeddings, text_embeddings = model(image.to(device), text.to(device))

        loss = contrastive_loss(image_embeddings, text_embeddings) # to avoid nan
        if torch.isnan(loss):
            continue

        pbar.set_description("[INFO] Loss %.4f" % loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()

    torch.save(model.state_dict(),'save_epoch_' + str(n) +".pth")
