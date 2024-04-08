import torch.nn as nn

from FPLF.FPB import FPB
from backbone.EfficientNetB4 import EfficientNetB4, switch_layers

class FPLF(nn.Module):
    def __init__(self, cfg_path):
        super(FPLF, self).__init__()
        self.backbone = EfficientNetB4(pretrained=True, num_classes=1, cfg_path=cfg_path)
        self.fpbList = nn.Sequential(
            FPB(dim=32, h=80, w=41, fp32fft=True),
            FPB(dim=32, h=80, w=41, fp32fft=True),
            FPB(dim=32, h=80, w=41, fp32fft=True),
            FPB(dim=32, h=80, w=41, fp32fft=True),
        )

    def forward(self, x):
        x = switch_layers(self.backbone, 'stem', x)
        x = switch_layers(self.backbone, 'blocks', x, 0, 1)
        x = self.fpbList(x)
        x = switch_layers(self.backbone, 'blocks', x, 2, 6)
        x = switch_layers(self.backbone, 'head', x)
        fea, x = switch_layers(self.backbone, 'fc', x)
        output = {"logits": x, "features": fea}
        return output
