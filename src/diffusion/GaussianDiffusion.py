import math
import torch
from torch import nn
from torch.functional import F
#%%
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        num_timesteps=1000,        # 扩散总步数
        beta_schedule="linear",    # beta调度方式: linear/cosine
        beta_start=0.0001,         # beta起始值
        beta_end=0.02,             # beta终止值
        loss_type="mse",           # 损失类型: mse (噪声预测) / l1 / huber
        objective="eps",           # 预测目标: eps (噪声) / x0 (原始图像)
        scale_shift_norm=False     # 是否使用scale-shift归一化
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.objective = objective
        self.scale_shift_norm = scale_shift_norm

        # 1. 定义beta调度
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            betas = self.cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # 2. 注册缓冲区 (不参与梯度更新)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1. - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine调度 (参考Improved DDPM)"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # -------------------------- 核心计算函数 --------------------------
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, model, x_start, t, y=None, noise=None):
        """计算训练损失"""
        # 1. 前向加噪
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        # 2. 模型预测
        model_output = model(x_t, t, y=y)

        # 3. 根据预测目标计算损失
        if self.objective == "eps":
            target = noise
        elif self.objective == "x0":
            target = x_start
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # 4. 损失计算
        if self.loss_type == "mse":
            loss = F.mse_loss(model_output, target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(model_output, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss.mean(dim=[1,2,3])  # 按样本平均

    def p_mean_variance(self, model, x, t, y=None):
        """计算逆过程均值和方差"""
        # 1. 模型预测
        model_output = model(x, t, y=y)

        # 2. 解析后验参数
        if self.objective == "eps":
            # 预测噪声 → 推导x0
            pred_xstart = self.predict_xstart_from_eps(x, t, model_output)
        elif self.objective == "x0":
            pred_xstart = model_output
        else:
            raise ValueError

        # 3. 计算后验均值方差
        model_mean, posterior_variance = self.q_posterior(pred_xstart, x, t)
        return {"mean": model_mean, "variance": posterior_variance, "pred_xstart": pred_xstart}

    def q_posterior(self, x_start, x_t, t):
        """计算后验分布 q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance

    # -------------------------- 工具函数 --------------------------
    def predict_xstart_from_eps(self, x_t, t, eps):
        """从预测的噪声推导x0"""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    @property
    def posterior_params(self):
        """预计算后验参数"""
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)

        # 后验方差计算
        posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # 后验均值系数
        posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        return posterior_variance, posterior_mean_coef1, posterior_mean_coef2

# 辅助函数：从缓冲区提取对应时间步的值
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape)-1)))

