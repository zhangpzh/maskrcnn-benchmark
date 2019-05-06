import torch
from torch import nn


class attentionMixupModule(nn.Module):
    def __init__(self, cfg):
        super(attentionMixupModule, self).__init__()
        # TODO(peizhen): remember to add config cfg.MODEL.ATTENTION.IN_CHANNELS (2048x2 or sth)
        self.in_channels = cfg.MODEL.ATTENTION.IN_CHANNELS
        self.pred_mixup = nn.Sequential(
            nn.Conv2d(int(self.in_channels), int(self.in_channels / 4), kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(self.in_channels / 4), int(self.in_channels / 16), kernel_size=1, stride=1),
            # nn.linear(in_channels/16, 2), # 8x2
            nn.Conv2d(int(self.in_channels / 16), 2, kernel_size=1, stride=1),  # 8x2x1x1
            nn.Softmax()
        )

    # stk_feat_map: (8,4096,*,*)
    # stk_orig_img: (8,6,H,W)
    # return merged_img: (8,3,H,W)
    def forward(self, stk_feat_map, stk_orig_img):
        x = self.pred_mixup(stk_feat_map)
        # (8,2,1,1) -> (8,2x3,1,1)
        indices = torch.LongTensor([0, 0, 0, 1, 1, 1])
        x = torch.index_select(x, 1, indices.to(torch.device('cuda')))
        # 与 8x6xHxW的原图broadcast乘法
        # (8,6,H,W)

        import ipdb
        ipdb.set_trace()

        mixup_img = x * stk_orig_img
        # 将8x6xHxW的上三层与下三层相加得到8x3xHxW返回
        weighted_img1, weighted_img2 = torch.split(mixup_img, [3, 3], dim=1)
        merged_img = weighted_img2 + weighted_img2
        return merged_img


def build_attentionMixup_module(cfg):
    return attentionMixupModule(cfg)
