"""Custom CLIP architecture featuring an additional fc layer"""

import clip
import torch


class CustomCLIP(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        model, _ = clip.load("RN50")

        # take the visual encoder of CLIP
        # we also convert it to be 32 bit (by default CLIP is 16)
        self.encoder = model.visual.float()

        # add a linear layer
        self.classifier = torch.nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)

        return x

