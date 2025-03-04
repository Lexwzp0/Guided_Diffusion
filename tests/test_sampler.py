import torch
import pytest
from torch.testing import assert_close

class TestGuidedDiffusionSampler:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化模拟组件
        class MockModel(nn.Module):
            def forward(self, x, t, y=None):
                return torch.randn_like(x)  # 返回随机噪声

        class MockClassifier(nn.Module):
            def forward(self, x, t):
                return torch.randn(x.size(0), 10)  # 10个类别

        class MockDiffusion:
            num_timesteps = 100
            def p_mean_variance(self, model, x, t, y=None):
                return {
                    "mean": x * 0.8,
                    "variance": torch.ones_like(x) * 0.1,
                    "pred_xstart": x * 0.5
                }

        self.model = MockModel().to(self.device)
        self.classifier = MockClassifier().to(self.device)
        self.diffusion = MockDiffusion()

        self.sampler = GuidedDiffusionSampler(
            diffusion_process=self.diffusion,
            model=self.model,
            classifier=self.classifier,
            classifier_scale=5.0
        )

        self.batch_size = 4
        self.img_shape = (self.batch_size, 1, 32, 32)

    # -------------------------- 核心功能测试 --------------------------
    def test_cond_fn_gradient(self, setup):
        """验证分类器梯度计算正确性"""
        x = torch.randn(*self.img_shape, device=self.device)
        t = torch.tensor([10, 20, 30, 40], device=self.device)
        y = torch.randint(0, 10, (self.batch_size,), device=self.device)

        grad = self.sampler.cond_fn(x, t, y)
        assert grad.shape == x.shape, "梯度形状错误"
        assert not torch.allclose(grad, torch.zeros_like(grad)), "梯度不应全零"

    def test_p_sample_with_guidance(self, setup):
        """验证带引导的采样步骤"""
        x = torch.randn(*self.img_shape, device=self.device)
        t = torch.full((self.batch_size,), 50, device=self.device)
        y = torch.randint(0, 10, (self.batch_size,), device=self.device)

        # 原始均值
        original_mean = self.diffusion.p_mean_variance(None, x, t)["mean"]

        # 引导后均值
        guided_sample = self.sampler.p_sample(x, t, y)
        assert guided_sample.shape == x.shape, "采样形状错误"
        assert not torch.allclose(guided_sample, original_mean), "均值应被调整"

    def test_full_sampling_loop(self, setup):
        """验证完整采样流程"""
        y = torch.randint(0, 10, (self.batch_size,), device=self.device)
        samples = self.sampler.p_sample_loop(self.img_shape, y)

        assert samples.shape == self.img_shape
        assert samples.min() >= -1.5 and samples.max() <= 1.5, "数值范围异常"

    # -------------------------- 边界条件测试 --------------------------
    def test_zero_scale_guidance(self, setup):
        """验证无分类器引导的情况"""
        zero_scale_sampler = GuidedDiffusionSampler(
            diffusion_process=self.diffusion,
            model=self.model,
            classifier=self.classifier,
            classifier_scale=0.0  # 关闭引导
        )

        x = torch.randn(*self.img_shape, device=self.device)
        t = torch.full((self.batch_size,), 50, device=self.device)
        y = torch.randint(0, 10, (self.batch_size,), device=self.device)

        original_mean = self.diffusion.p_mean_variance(None, x, t)["mean"]
        guided_sample = zero_scale_sampler.p_sample(x, t, y)

        # 应接近原始均值 + 噪声
        expected = original_mean + torch.exp(0.5 * torch.log(torch.tensor(0.1))) * torch.randn_like(x)
        assert_close(guided_sample, expected, rtol=0.1, atol=0.1)

    def test_batch_consistency(self, setup):
        """验证不同批次的采样结果不同"""
        y = torch.randint(0, 10, (self.batch_size,), device=self.device)
        samples1 = self.sampler.p_sample_loop(self.img_shape, y)
        samples2 = self.sampler.p_sample_loop(self.img_shape, y)

        assert not torch.allclose(samples1, samples2, atol=1e-3), "应存在随机性"

    # -------------------------- 生成测试 --------------------------
    def test_generate_output(self, setup):
        """验证生成函数的输出规范"""
        num_samples = 5
        generated = self.sampler.generate(
            num_samples=num_samples,
            num_classes=10,
            image_size=(32, 32)
        )

        assert generated.shape[0] == num_samples, "生成数量错误"
        assert generated.min() >= 0 and generated.max() <= 1, "数值未正确归一化"

if __name__ == "__main__":
    pytest.main(["-v", "-s"])