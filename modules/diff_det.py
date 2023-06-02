# Algorithm 1 DiffusionDet Training
import os.path
import pickle
import shutil
from inspect import isfunction

import numpy as np
import torch
from clip import clip
from torch import nn
from torch.utils.data import random_split
import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from utilities import resize_bbox
from refcocog import RefCOCOgSample, RefCOCOg


# Layers strucure taken from:
# https://huggingface.co/blog/annotated-diffusion

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


# The SinusoidalPositionEmbeddings module takes a tensor of shape (batch_size, 1) as input (i.e. the noise levels of several noisy images in a batch), and turns this into a tensor of shape (batch_size, dim), with dim being the dimensionality of the position embeddings. This is then added to each residual block, as we will see further.

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, out_channels=init_dim, kernel_size=1,
                                   padding=0)  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class DiffClip(nn.Module):
    def __init__(self, time_steps=100, clip_ver="RN101", device='cuda', *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.time = time_steps
        self.clip_model, self.clip_prep = clip.load(clip_ver, device=device)

        in_dim = 40  # clip output
        self.unet = Unet(dim=40, init_dim=in_dim, channels=1)
        self.unet.to(device)

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=self.time)

        # define alphas
        self.alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-3)
        self.bbox_mean = torch.tensor((0,0,0,0)).to(device)
        self.bbox_std = torch.tensor((1,1,1,1)).to(device)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def normalize_bbox(self, bbox, reverse=False) -> tuple:
        """
        """
        bbox = torch.tensor(bbox).to(device)
        if not reverse:
            bbox = (bbox - self.bbox_mean) / self.bbox_std

            return bbox
        else:
            bbox = bbox *  self.bbox_std + self.bbox_mean

            return bbox.tolist()[0]

    def _encode_text(self, text):
        text_ = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_text(text_)

    def _encode_img(self, image):
        image_ = self.clip_prep(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_image(image_)

    def encoder(self, raw_image, text, bbox):
        """ Produces an encoded tensor"""
        image = self._encode_img(raw_image)
        text = self._encode_text(text)
        if bbox is None:
            bbox_enc = torch.normal(mean=0, std=1, size=(1, 512 + 64)).to(self.device)
        else:
            bbox_enc = torch.zeros(1, 512 + 64).to(self.device)
            bbox_enc[0, 0:4] = self.normalize_bbox(bbox)
        encoding = torch.cat((image, text, bbox_enc), 1).reshape(40, 40)
        return encoding

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        # we should pass the noise
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.zeros((len(x_start), 40, 40)).to(device)
            # noise[:, 25:26, 24:28] = x_start[:, 25:26, 24:28] + torch.randn_like(x_start[:, 25:26, 24:28]) # noise of bb true
            # noise[:,26:,28:] += torch.randn_like(x_start[:,26:,28:]) # noise of bb pad
            noise[:, 25:, 28:] += torch.randn_like(x_start[:, 25:, 28:])  # noise of bb pad

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = x_noisy[None, :, :, :].movedim(1, 0)
        noise = noise[None, :, :, :].movedim(1, 0)
        predicted_noise = self.unet(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def p_sample(self, x, t, t_index):
        with torch.no_grad():
            betas_t = self.extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = self.extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

            # Equation 11 in the paper
            # Use our model (noise predictor) to predict the mean
            model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * self.unet(x, t) / sqrt_one_minus_alphas_cumprod_t
            )

            if t_index == 0:
                return model_mean
            else:
                posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
                # noise = torch.zeros(3,512).to(self.device)
                noise = torch.zeros((len(x), 40, 40)).to(device)
                noise = noise[None, :, :, :].movedim(1, 0)
                noise[:, :, 25:, 28:] += torch.randn_like(x[:, :, 25:, 28:])  # noise of bb pad
                # Algorithm 2 line 4:
                return model_mean + torch.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, images, shape):
        with torch.no_grad():
            device = next(self.unet.parameters()).device

            b = shape[0]
            # start from pure noise BB (for each example in the batch), as initialized in the encoder

            ris = None

            for i in tqdm(reversed(range(0, self.time)), desc='sampling loop time step', total=self.time):
                img = self.p_sample(images, torch.full((b,), i, device=device, dtype=torch.long), i)
                ris = img
            return ris

    def sample(self, images, image_size, batch_size=16, channels=1):
        with torch.no_grad():
            # add dummy channel
            images = images[None, :, :, :].movedim(1, 0)
            return self.p_sample_loop(images, shape=(batch_size, channels, image_size, image_size))


if __name__ == "__main__":
    from utilities import IoU
    print("Loading dataset")
    data_path = "/media/dmmp/vid+backup/Data/refcocog"
    # data_path = "dataset/refcocog"
    train_ds = RefCOCOg(ds_path=data_path, split='train')
    save_path = os.path.normpath(os.path.join("saved_models", "diff_clip"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = RefCOCOg(ds_path=data_path)
    # keep only a toy portion of each split
    batch_size = 256
    keep = 1
    train = False
    red_dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
    if train:
        print("Instantiating model")
        net = DiffClip()
        red_train_ds, _ = random_split(train_ds, [int(keep * len(train_ds)), len(train_ds) - int(keep * len(train_ds))])
        train_loader = torch.utils.data.DataLoader(red_train_ds, batch_size=batch_size, shuffle=True,
                                                   collate_fn=lambda x: x)
        net.batches = len(train_loader)

        # calculate bboxes mean and std
        bboxes = []
        print("calculating bboxes mean and std")
        for batch in tqdm(train_loader):
            for el in batch:
                bboxes.append(el['bbox'])
                pass
        bboxes = torch.tensor(bboxes)
        net.bbox_mean = torch.mean(bboxes, dim=0).to(net.device)
        net.bbox_std = torch.std(bboxes, dim=0).to(net.device)
        del bboxes


        device = 'cuda'
        epochs = 30
        for epoch in tqdm(range(epochs)):
            for step, batch in tqdm(enumerate(train_loader)):
                print("Extracting batch tensors", flush=True)
                batch_elements = []
                for el in batch:
                    for sentence in el['sentences']:
                        batch_elements.append(net.encoder(el['img'], sentence, el['bbox']))

                batch = torch.stack(batch_elements)
                print("Starting batch diffusion", flush=True)
                net.optimizer.zero_grad()

                batch_size = len(batch_elements)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, net.time, (batch_size,), device=device).long()

                loss = net.p_losses(batch, t, loss_type="huber")

                if step % 20 == 0:
                    print("Loss:", loss.item(), flush=True)

                loss.backward()
                net.optimizer.step()

            pass
            print("Saving epoch model")
            path = os.path.normpath(os.path.join(save_path, "diff_clip_epoch_" + str(epoch) + "|" + str(loss.item())))
            with open(path, 'wb') as f:
                pickle.dump(net, f)
                print("Model saved as: " + path)

        print("Saving best model")
        files = os.listdir(save_path)
        losses = []
        for f in files:
            losses.append(f.split('|')[-1])
        best_loss_id = losses.index(min(losses))
        s = os.path.normpath(os.path.join(save_path, files[best_loss_id]))
        d = os.path.normpath(os.path.join(save_path, "best_model.pickle"))

        shutil.copyfile(src=s, dst=d)
        print(s + " has been saved as " + d)

    else:
        red_test_ds, _ = random_split(train_ds, [int(keep * len(train_ds)), len(train_ds) - int(keep * len(train_ds))])
        test_loader = torch.utils.data.DataLoader(red_test_ds, batch_size=batch_size, shuffle=True,
                                                  collate_fn=lambda x: x)
        print("Instantiating model")
        model_path = os.path.normpath(os.path.join(save_path, 'best_model.pickle'))
        with open(model_path, 'rb') as f:
            net = pickle.load(f)
        net.batches = len(test_loader)
        device = 'cuda'
        average_iou = 0
        iou = 0
        counter = 0
        true_bboxes = []
        for step, batch in tqdm(enumerate(test_loader)):
            print("Extracting batch tensors", flush=True)
            batch_elements = []
            for el in batch:
                for sentence in el['sentences']:
                    batch_elements.append(net.encoder(el['img'], sentence, None))
                    true_bboxes.append(el['bbox'])

            batch = torch.stack(batch_elements)
            print("Starting batch inference (t = " + str(net.time) + ")", flush=True)
            samples = net.sample(batch, image_size=40, batch_size=len(batch), channels=1)
            for i, sample in enumerate(samples):
                counter +=1
                predicted_bb = net.normalize_bbox(sample[0, 25:26, 24:28], reverse=True)
                true_bb = true_bboxes[i]
                iou += IoU(true_bb, predicted_bb)
                average_iou = iou / counter
                pass
            print("Average IoU: ", average_iou)

