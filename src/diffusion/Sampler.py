import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class GuidedDiffusionSampler:
    def __init__(self,
                 diffusion_process,
                 model,
                 classifier,
                 classifier_scale=1.0,
                 ddim=False):
        """
        Args:
            diffusion_process: 预定义的扩散过程（如GaussianDiffusion）
            model: 你的ConditionalDiffusionUNet实例
            classifier: 训练好的噪声鲁棒分类器
            classifier_scale: 分类器引导强度系数
            ddim: 是否使用DDIM加速采样
        """
        self.diffusion = diffusion_process
        self.model = model
        self.classifier = classifier
        self.scale = classifier_scale
        self.ddim = ddim

    def cond_fn(self, x, t, y=None):
        """分类器梯度计算函数"""
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.scale

    def model_fn(self, x, t, y=None):
        """包装UNet前向传播"""
        return self.model(x, t, y=y)

    def p_sample(self, x, t, y):
        """单步采样（带分类器引导）"""
        # 原始扩散参数
        out = self.diffusion.p_mean_variance(self.model, x, t, y=y)

        # 应用分类器梯度
        if self.scale > 0:
            grad = self.cond_fn(x, t, y)
            out["mean"] = out["mean"] + out["variance"] * grad

        # 重参数化采样
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1]*(len(x.shape)-1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return sample

    def p_sample_loop(self, shape, y):
        """完整采样循环"""
        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)

        for t in reversed(range(0, self.diffusion.num_timesteps)):
            timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, y=y)

        return x

    def generate(self, num_samples, num_classes, image_size):
        """生成入口函数"""
        self.model.eval()
        self.classifier.eval()

        all_images = []
        with torch.no_grad():
            for _ in range((num_samples-1)//self.batch_size + 1):
                # 生成目标类别标签
                y = torch.randint(0, num_classes, (self.batch_size,), device=device)

                # 执行采样
                samples = self.p_sample_loop(
                    shape=(self.batch_size, 1, image_size[0], image_size[1]),
                    y=y
                )

                # 后处理
                samples = (samples.clamp(-1, 1) + 1) / 2  # 缩放到[0,1]
                all_images.append(samples.cpu())

        return torch.cat(all_images, dim=0)[:num_samples]