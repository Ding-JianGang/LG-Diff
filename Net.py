import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.diffNAFNet import ConditionalNAFNet


def data_transform(X):
    # return 2 * X - 1.0
    return  X


def inverse_data_transform(X):
    # return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
    return X

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()

        # self.args = args
        # self.config = config
        # self.device = config.device

        # self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        # self.high_enhance1 = HFRM(in_channels=3, out_channels=64)
        self.Unet = ConditionalNAFNet()

        # betas = get_beta_schedule(
        #     beta_schedule=config.diffusion.beta_schedule,
        #     beta_start=config.diffusion.beta_start,
        #     beta_end=config.diffusion.beta_end,
        #     num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        # )

        betas = get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=200,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        # skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        # seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        skip = 200 // 10
        seq = range(0, 200, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=x_cond.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(xt,x_cond, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}
        # dwt, idwt = DWT(), IWT()

        input_img = x[:, :3, :, :]
        # n, c, h, w = input_img.shape
        input_img = data_transform(input_img)
        # input_img_norm = data_transform(input_img)
        # input_dwt = dwt(input_img_norm)

        # input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        # input_high0 = self.high_enhance0(input_high0)

        # input_LL_dwt = dwt(input_LL)
        # input_LL_LL, input_high1 = input_LL_dwt[:n, ...], input_LL_dwt[n:, ...]
        # input_high1 = self.high_enhance1(input_high1)

        b = self.betas.to(input_img.device)

        # t = torch.randint(low=0, high=self.num_timesteps, size=(input_img.shape[0] // 2 + 1,)).to(input_img.device)
        # t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_img.shape[0]].to(x.device)
        # a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        # e = torch.randn_like(input_img)

        if self.training:
            # gt_img_norm = data_transform(x[:, 3:, :, :])
            # gt_dwt = dwt(gt_img_norm)
            # gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]
            #
            # gt_LL_dwt = dwt(gt_LL)
            # gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]
            # gt_LL_LL = data_transform(x[:, 3:, :, :])

            input_LL_LL = input_img

            # x_t = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            # noise_output = self.Unet(x_t,input_LL_LL, t.float())
            denoise_LL_LL = self.sample_training(input_LL_LL, b)
            pred_x = denoise_LL_LL

            # pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))

            # pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            # data_dict["input_high0"] = input_high0
            # data_dict["input_high1"] = input_high1
            # data_dict["gt_high0"] = gt_high0
            # data_dict["gt_high1"] = gt_high1
            # data_dict["pred_LL"] = pred_LL
            # data_dict["gt_LL"] = gt_LL
            # data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            # data_dict["e"] = e
            data_dict["gt_x"] = x[:, 3:, :, :]

        else:
            input_LL_LL = input_img
            denoise_LL_LL = self.sample_training(input_LL_LL, b)
            pred_x = denoise_LL_LL
            # pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))
            # pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["pred_x"] = pred_x

        return data_dict
# from thop import profile
# net = ConditionalNAFNet().cuda()
# input = torch.randn(1,3, 256,256).cuda()
# t = torch.randn(1).cuda() # batchsize=1, 输入向量长度为10
# macs, params = profile(net, inputs=(input,input,t))
# print(' FLOPs: ', macs*2)   # 一般来讲，FLOPs是macs的两倍
# print('params: ', params)