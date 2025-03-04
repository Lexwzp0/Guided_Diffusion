import pytest
import torch
import torch.nn as nn
from src.diffusion.ConditionalDiffusion import ConditionalDiffusionUNet
from torch.testing import assert_close
chs = [1, 64, 128, 256]

class TestConditionalDiffusionUNet:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 模拟必要的组件
        class ConditionalEmbedding(nn.Module):
            def __init__(self, num_classes, time_dim, label_dim):
                super().__init__()
                self.time_embed = nn.Embedding(num_classes, time_dim)
                self.label_embed = nn.Embedding(num_classes, label_dim)

            def forward(self, t, y):
                return self.time_embed(t) + self.label_embed(y)

        class ConditionalBlock(nn.Module):
            def __init__(self, in_ch, out_ch, cond_dim):
                super().__init__()
                self.main = nn.Conv2d(in_ch, out_ch, 3, padding=1)
                self.cond_proj = nn.Linear(cond_dim, out_ch * 2)

            def forward(self, x, cond):
                gamma, beta = self.cond_proj(cond).chunk(2, 1)
                return self.main(x) * gamma[..., None, None] + beta[..., None, None]

        # 使用简化版组件替代原始实现
        self.model = ConditionalDiffusionUNet(
            num_classes=10,
            time_dim=128,
            label_dim=64
        ).to(self.device)

        # 替换自定义组件
        self.model.cond_embed = ConditionalEmbedding(10, 128, 64)
        self.model.down = nn.ModuleList([
            nn.Sequential(
                ConditionalBlock(chs[i], chs[i + 1], 128),
                ConditionalBlock(chs[i + 1], chs[i + 1], 128),
            ) for i in range(3)
        ])

        # 测试参数
        self.batch_size = 4
        self.input_shape = (24, 50)  # 原始输入尺寸 (H, W)

    # -------------------------- 核心功能测试 --------------------------
    def test_io_shape(self, setup):
        """验证输入输出形状正确性"""
        x = torch.randn(self.batch_size, *self.input_shape, device=self.device)
        t = torch.randint(0, 1000, (self.batch_size,), device=self.device)
        y = torch.randint(0, 10, (self.batch_size,), device=self.device)

        output = self.model(x, t, y)
        assert output.shape == (self.batch_size, *self.input_shape), \
            f"期望输出形状 {(self.batch_size, *self.input_shape)}，实际得到 {output.shape}"

    def test_device_consistency(self, setup):
        """验证所有张量设备一致性"""
        x = torch.randn(2, 24, 50).to(self.device)
        t = torch.randint(0, 1000, (2,)).to(self.device)
        y = torch.randint(0, 10, (2,)).to(self.device)

        output = self.model(x, t, y)
        assert output.device == x.device, "输出设备不一致"
        for param in self.model.parameters():
            assert param.device == x.device, "模型参数设备不一致"

    def test_gradient_flow(self, setup):
        """验证梯度传播正常"""
        x = torch.randn(2, 24, 50, device=self.device, requires_grad=True)
        t = torch.randint(0, 1000, (2,), device=self.device)
        y = torch.randint(0, 10, (2,), device=self.device)

        output = self.model(x, t, y)
        loss = output.mean()
        loss.backward()

        # 检查关键层梯度
        assert x.grad is not None, "输入梯度未传播"
        assert self.model.down[0][0].main.weight.grad is not None, "下采样层梯度未更新"
        assert self.model.cond_embed.label_embed.weight.grad is not None, "条件嵌入梯度未更新"

    def test_condition_effectiveness(self, setup):
        """验证条件嵌入有效性"""
        x = torch.randn(1, 24, 50, device=self.device)
        t = torch.tensor([500], device=self.device)

        # 相同输入不同标签
        y1 = torch.tensor([3], device=self.device)
        y2 = torch.tensor([7], device=self.device)

        out1 = self.model(x, t, y1)
        out2 = self.model(x, t, y2)

        assert not torch.allclose(out1, out2, atol=1e-3), "不同标签应产生不同输出"

    # -------------------------- 边界条件测试 --------------------------
    def test_extreme_inputs(self, setup):
        """处理极端输入不崩溃"""
        # 全零输入
        x_zero = torch.zeros(1, 24, 50, device=self.device)
        t = torch.tensor([999], device=self.device)
        y = torch.tensor([5], device=self.device)
        output = self.model(x_zero, t, y)
        assert torch.isfinite(output).all(), "输出包含非法值"

        # 随机大值输入
        x_large = torch.randn(1, 24, 50, device=self.device) * 100
        output = self.model(x_large, t, y)
        assert output.shape == (1, 24, 50)

    def test_edge_batch_size(self, setup):
        """测试边界批次大小"""
        # 空批次（应报错）
        with pytest.raises(RuntimeError):
            x = torch.randn(0, 24, 50, device=self.device)
            t = torch.tensor([], dtype=torch.long, device=self.device)
            y = torch.tensor([], dtype=torch.long, device=self.device)
            self.model(x, t, y)

        # 超大批次
        x = torch.randn(1024, 24, 50, device=self.device)
        t = torch.randint(0, 1000, (1024,), device=self.device)
        y = torch.randint(0, 10, (1024,), device=self.device)
        output = self.model(x, t, y)
        assert output.shape[0] == 1024


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
