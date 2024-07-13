# LG-Diff
Learning to Follow Local Class-Regional Guidance for Nearshore Image Cross-Domain High-Quality Translation
This is the PyTorch implementation of the color-to-thermal image translation. The code is based on the PyTorch implementation of the Diffusers (https://github.com/huggingface/diffusers) and DiffIR (https://github.com/Zj-BinXia/DiffIR).

## Prerequisites
    Linux or Win10 
    Python ≥ 3.9 
    NVIDIA GPU + CuDNN CUDA ≥ 11.3
    GPU memory ≥ 40G

## Dataset

## Demo-
Demo- files is used to verify the effectiveness of the local class region guidance strategy on diffusers. It can be trained directly.

## Aligned and unaligned videos
For image translation tasks, unaligned video streams can only be achieved in an unsupervised or non-regression manner. In contrast, for regression models to perform better, it is usually required that cross-modal video streams appear in pairs.
* An example of an unaligned video stream.
![An example of an unaligned video stream](https://github.com/Ding-JianGang/LG-Diff/blob/main/image/unalign.gif)
* An example of an aligned video stream.
![An example of an aligned video stream](https://github.com/Ding-JianGang/LG-Diff/blob/main/image/align.gif)

## Result
* Visual comparisons on unknown challenging instances from InfraredCoast.
![Visual comparisons on unknown challenging instances from InfraredCoast.](https://github.com/Ding-JianGang/LG-Diff/blob/main/image/Result.jpg)
* Intuitive cases to illustrate the importance of local class-regional guidance.
![Intuitive cases to illustrate the importance of local class-regional guidance.](https://github.com/Ding-JianGang/LG-Diff/blob/main/image/Feature.jpg)

## Code download link for the control group
AttentionGAN: Training and Testing are followed by https://github.com/Ha0Tang/AttentionGAN.

Pix2Pix: Training and Testing are followed by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

CycleGAN: Training and Testing are followed by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

BicycleGAN: Training and Testing are followed by https://github.com/junyanz/BicycleGAN.

GCGAN: Training and Testing are followed by https://github.com/hufu6371/GcGAN.

DCLGAN: Training and Testing are followed by https://github.com/JunlinHan/DCLGAN.

CUT: Training and Testing are followed by https://github.com/taesungp/contrastive-unpaired-translation.

UNIT: Training and Testing are followed by https://github.com/mingyuliutw/UNIT.

MUNIT: Training and Testing are followed by https://github.com/NVlabs/MUNIT.

DRIT: Training and Testing are followed by https://github.com/HsinYingLee/DRIT.

MSGAN: Training and Testing are followed by https://github.com/HelenMao/MSGAN.

Conditional-GAN: Training and Testing are followed by https://github.com/huggingface/diffusers.
