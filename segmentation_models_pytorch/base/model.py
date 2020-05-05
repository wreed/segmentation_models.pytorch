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
        # print(x.shape)
        # x = x.permute(0, 2, 3, 1)
        # print(x.shape)
        x = x / 255.
        # x = x - mean
        # x = x / std
        for xi in range(3):
            x[0][xi] = x[0][xi] - self.mean[xi]
            x[0][xi] = x[0][xi] / self.std[xi]
        # x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        return x

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # import pdb;pdb.set_trace()
        x = self.normalize(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        masks = masks * 255.

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
