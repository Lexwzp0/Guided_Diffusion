#%%
import torch
import torch.nn as nn
from torch.testing import assert_close
from GaussianDiffusion import GaussianDiffusion
import pytest

class TestGaussianDiffusion:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 4
        self.img_size = (3, 32, 32)
        self.x_start = torch.randn(self.batch_size, *self.img_size).to(self.device)

        # 初始化两种调度方式的扩散模型
        self.diffusion_linear = GaussianDiffusion(
            num_timesteps=1000,
            beta_schedule="linear",
            objective="eps"
        ).to(self.device)

        self.diffusion_cosine = GaussianDiffusion(
            num_timesteps=1000,
            beta_schedule="cosine",
            objective="x0"
        ).to(self.device)

    # -------------------------- 基础参数校验 --------------------------
    def test_parameter_shapes(self, setup):
        """验证所有缓存参数的形状一致性"""
        for diffusion in [self.diffusion_linear, self.diffusion_cosine]:
            assert diffusion.betas.shape == (diffusion.num_timesteps,)
            assert diffusion.alphas.shape == (diffusion.num_timesteps,)
            assert diffusion.alphas_cumprod.shape == (diffusion.num_timesteps,)

    def test_beta_range(self, setup):
        """验证beta值范围合理性"""
        for diffusion in [self.diffusion_linear, self.diffusion_cosine]:
            assert torch.all(diffusion.betas > 0), "存在非正beta值"
            assert torch.all(diffusion.betas < 1), "存在超过1的beta值"
            assert torch.all(diffusion.betas[:-1] <= diffusion.betas[1:]), "beta序列非单调递增"

    # -------------------------- 前向过程校验 --------------------------
    def test_q_sample_consistency(self, setup):
        """验证加噪过程确定性部分的一致性"""
        t = torch.tensor([50, 100, 200, 500], device=self.device)
        noise = torch.randn_like(self.x_start)

        # 两次相同噪声输入应得到相同结果
        x_t1 = self.diffusion_linear.q_sample(self.x_start, t, noise)
        x_t2 = self.diffusion_linear.q_sample(self.x_start, t, noise)
        assert_close(x_t1, x_t2, rtol=1e-4, atol=1e-4)

    def test_q_sample_statistics(self, setup):
        """验证加噪结果的统计特性"""
        t = torch.full((self.batch_size,), 500, device=self.device)
        x_t = self.diffusion_linear.q_sample(self.x_start, t)

        # 计算理论均值方差
        sqrt_alpha = self.diffusion_linear.sqrt_alphas_cumprod[500]
        sqrt_one_minus_alpha = self.diffusion_linear.sqrt_one_minus_alphas_cumprod[500]
        theoretical_mean = sqrt_alpha * self.x_start
        theoretical_var = (sqrt_one_minus_alpha ** 2) * torch.ones_like(self.x_start)

        # 允许5%的误差范围
        assert_close(x_t.mean(), theoretical_mean.mean(), rtol=0.05)
        assert_close(x_t.var(), theoretical_var.mean(), rtol=0.05)

    # -------------------------- 损失计算校验 --------------------------
    def test_loss_computation(self, setup):
        """验证不同预测目标下的损失计算正确性"""
        # 模拟一个简单模型
        class DummyModel(nn.Module):
            def forward(self, x, t, y=None):
                if self.diffusion.objective == "eps":
                    return torch.randn_like(x)  # 预测噪声
                else:
                    return x  # 预测x0

        # 测试EPS模式
        dummy_model = DummyModel().to(self.device)
        t = torch.randint(0, 1000, (self.batch_size,), device=self.device)
        loss = self.diffusion_linear.p_losses(dummy_model, self.x_start, t)
        assert loss.shape == (self.batch_size,), "损失形状错误"

        # 测试X0模式
        loss_x0 = self.diffusion_cosine.p_losses(dummy_model, self.x_start, t)
        assert not torch.allclose(loss, loss_x0), "不同目标模式的损失应不同"

    # -------------------------- 后验计算校验 --------------------------
    def test_posterior_mean(self, setup):
        """验证后验均值计算的正确性"""
        t = torch.tensor([500], device=self.device)
        x_t = self.diffusion_linear.q_sample(self.x_start, t)

        # 构造完美预测噪声的模型
        class PerfectModel(nn.Module):
            def forward(self, x, t, y=None):
                return self.diffusion.predict_xstart_from_eps(x, t, (x_t - self.diffusion.sqrt_alphas_cumprod[t] * self.x_start) / self.diffusion.sqrt_one_minus_alphas_cumprod[t])

        model = PerfectModel()
        model.diffusion = self.diffusion_linear

        # 计算后验参数
        out = self.diffusion_linear.p_mean_variance(model, x_t, t)
        assert_close(out["pred_xstart"], self.x_start, rtol=1e-3, atol=1e-3)

    # -------------------------- 设备兼容性校验 --------------------------
    def test_device_consistency(self, setup):
        """验证所有缓存参数与模型设备一致"""
        for diffusion in [self.diffusion_linear, self.diffusion_cosine]:
            params = [diffusion.betas, diffusion.alphas, diffusion.alphas_cumprod]
            for param in params:
                assert param.device == torch.device(self.device), f"参数设备不一致: {param.device} vs {self.device}"

    # -------------------------- 边界条件校验 --------------------------
    def test_edge_timesteps(self, setup):
        """验证t=0和t=max_timestep时的行为"""
        # t=0 应接近原始数据
        t0 = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        x_t0 = self.diffusion_linear.q_sample(self.x_start, t0)
        assert_close(x_t0, self.x_start * self.diffusion_linear.sqrt_alphas_cumprod[0], rtol=1e-4)

        # t=max 应接近纯噪声
        t_max = torch.full((self.batch_size,), 999, device=self.device)
        x_tmax = self.diffusion_linear.q_sample(self.x_start, t_max)
        assert torch.allclose(x_tmax.mean(), torch.tensor(0.0, device=self.device), atol=0.1)
        assert torch.allclose(x_tmax.std(), torch.tensor(1.0, device=self.device), atol=0.1)

if __name__ == "__main__":
    pytest.main(["-v", "-s"])