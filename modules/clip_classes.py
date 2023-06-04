import clip
import torch


class CustomCLIP(torch.nn.Module):
    def __init__(self, num_classes: int = 10, w="RN50"):
        super().__init__()
        model, _ = clip.load(w)
        model = model.float()

        # Take the visual encoder of CLIP
        # We also convert it to be 32 bit (by default CLIP is 16)
        self.encoder = model.visual.float()
        self.txt_encoder = {
            'token_embedding': model.token_embedding,
            'positional_embedding': model.positional_embedding,
            'transformer': model.transformer,
            'ln_final': model.ln_final,
            'text_projection': model.text_projection

        }

        # Add a linear layer
        self.classifier = torch.nn.Linear(1024, num_classes)

    def encode_image(self, img):
        return self.encoder(img)

    def encode_text(self, text):
        x = self.txt_encoder['token_embedding'](text).type(torch.float32)  # [batch_size, n_ctx, d_model]

        x = x + self.txt_encoder['positional_embedding'].type(torch.float32)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.txt_encoder['transformer'](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.txt_encoder['ln_final'](x).type(torch.float32)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.txt_encoder['text_projection']

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)

        return x
