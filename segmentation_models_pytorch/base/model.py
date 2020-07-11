import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def normalize(self, x):

        x = x / 255.

        r = (x[0][0] - self.mean[0]) / self.std[0]
        g = (x[0][1] - self.mean[1]) / self.std[1]
        b = (x[0][2] - self.mean[2]) / self.std[2]

        x_norm = torch.unsqueeze(torch.stack((r,g,b), 0), 0)

        return x_norm

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        x = self.normalize(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        masks = torch.unsqueeze(torch.flatten(torch.squeeze(masks)), -1)
        # masks = masks * 255.

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
