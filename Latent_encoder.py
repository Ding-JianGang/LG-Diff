import torch
import torch.nn as nn


class latent_encoder_gelu(nn.Module):

    def __init__(self, in_chans=6, embed_dim=60, block_num=4, stage=1, groups=3):
        super(latent_encoder_gelu, self).__init__()

        # assert in_chans == int(6 // stage), "in chanel size is wrong"

        self.groups = groups

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans * 16, embed_dim, 3, 1, 1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=self.groups),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=self.groups))
            self.blocks.append(block)

        self.end = nn.Sequential(
            nn.Conv2d(embed_dim, 3, 3, 1, 1, groups=self.groups),
            nn.GELU(), )

    def forward(self, inp_img, gt=None):
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1)
        else:
            x = inp_img

        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.end(x)
        return x

class latent_encoder_gelu2(nn.Module):

    def __init__(self, in_chans=60, embed_dim=60, block_num=4, stage=1, groups=3):
        super(latent_encoder_gelu2, self).__init__()

        # assert in_chans == int(6 // stage), "in chanel size is wrong"

        self.groups = groups

        # self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 1, 1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=self.groups),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=self.groups))
            self.blocks.append(block)

        self.end = nn.Sequential(
            nn.Conv2d(embed_dim, 3, 3, 1, 1, groups=self.groups),
            nn.GELU(), )

    def forward(self, inp_img, gt=None):
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1)
        else:
            x = inp_img

        # x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.end(x)
        return x

# input = torch.randn(1,6,256,256)
# model = latent_encoder_gelu()
# # out = model(input)
#
#
# from thop import profile
# macs, params = profile(model, inputs=(input,))
#
# print("Number of FLOPs: {:.2f} G".format(macs / 1e9))
# print("Number of parameters: {:.2f} M".format(params / 1e6))


