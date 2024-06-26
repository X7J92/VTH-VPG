from torch import nn
import torch.nn.functional as F
from lib.models.map_modules import get_padded_mask_and_weight
from IPython import embed

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        kernel_sizes = cfg.KERNEL_SIZES
        strides = cfg.STRIDES
        paddings = cfg.PADDINGS
        dilations = cfg.DILATIONS
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size]+hidden_sizes
        # different due to conv3d
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv3d(channel_sizes[i], channel_sizes[i+1], k, s, p, d))

    def forward(self, x, mask):
        # x:(b,512,64,64) mask:(b,1,64,64)
        padded_mask = mask
        # convs: padding16,0,0,0, kernel9,9,9,9
        for i, pred in enumerate(self.convs):
            x = F.relu(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        return x