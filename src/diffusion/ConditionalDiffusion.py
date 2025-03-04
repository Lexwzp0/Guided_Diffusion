import torch
import torch.nn as nn
from data.pyg_dataToGraph import DataToGraph
import torch.nn.functional as F
from matplotlib import pyplot as plt
#%% md
### 热力图
#%%
def heatmap(numpy_array):
    # 绘制热力图
    plt.figure(figsize=(10, 5))
    plt.imshow(numpy_array, cmap='viridis', aspect='auto', vmin=-1, vmax=1)  # 调整颜色范围以更好地显示小值
    plt.colorbar(label='Value', extend='max')  # 颜色条
    plt.title('Tensor Visualization (Heatmap)')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.xticks(range(0, numpy_array.shape[1], 5))  # 设置 x 轴刻度
    plt.yticks(range(0, numpy_array.shape[0], 1))  # 设置 y 轴刻度（密集）
    plt.ylim(0, numpy_array.shape[0] - 1)  # 调整 y 轴范围

    # 绘制网格线
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()
#%%
# TODO 加载数据集
dataset = DataToGraph(
    raw_data_path='../../data/',
    dataset_name='TFF' + '.mat')  # 格式: [(graph,label),...,(graph,label)]

input_dim = dataset[0].x.size(1)
num_classes = dataset.num_classes

# 提取所有的x和y
x0 = []
labels = []

for data in dataset:
    # 提取x (形状为 [num_nodes, input_dim])
    # 但是你提到dataset.x的形状是 [24,50]，这可能是一个图的x特征矩阵
    x0.append(data.x)
    # 提取y（标量标签）
    labels.append(data.y)

# 将列表转换为张量
x0 = torch.stack(x0)  # 形状 [num_samples, 24, 50]
labels = torch.stack(labels)  # 形状 [num_samples]

print(num_classes)
print("X0 shape:", x0.shape)
print("Labels shape:", labels.shape)
#%%
# 将数据传输到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO 确定超参数的值
# 超参数值
num_steps = 1000  # 假设扩散步数为 1000
eps = 1e-5  # 避免除以零或引入过小的数值的小偏移量

# 生成时间步的序列
t = torch.linspace(0, 1, num_steps + 1)  # 主要时间步范围从 0 到 1

# 使用余弦调度生成 betas
betas = torch.cos(torch.pi / 2.0 * t) ** 2  # 余弦平方函数
betas = betas / betas.max()  # 归一化到 0-1 范围
betas = torch.flip(betas, [0])  # 反转顺序，以确保从小到大递增
betas = torch.clamp(betas, min=1e-5, max=0.5e-2)  # 调整范围到 (1e-5, 0.5e-2)

# 计算 alpha , alpha_prod , alpha_prod_previous , alpha_bar_sqrt 等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0)  # 累积连乘
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # p means previous
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

# 将超参数也移动到GPU
betas = betas.to(device)
alphas = alphas.to(device)
alphas_prod = alphas_prod.to(device)
alphas_prod_p = alphas_prod_p.to(device)
alphas_bar_sqrt = alphas_bar_sqrt.to(device)
one_minus_alphas_bar_log = one_minus_alphas_bar_log.to(device)
one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)

assert alphas_prod.shape == alphas_prod.shape == alphas_prod_p.shape \
       == alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
       == one_minus_alphas_bar_sqrt.shape
print("all the same shape:", betas.shape)
#%%
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
#%%
def exists(x):
    return x is not None
#%% md
### MySequential
#%%
class MySequential(nn.Sequential):
    def forward(self, x, t_emb):
        for module in self:
            if isinstance(module, ConditionalBlock):  # 仅对特定模块传参
                x = module(x, t_emb)
            elif isinstance(module, MyBlock):
                x = module(x, t_emb)
            else:  # 其他模块按默认方式处理
                x = module(x)
        return x

#%% md
### TimeEmbedding
#%%
class EnhancedTimeEmbedding(nn.Module):
    """增强时间嵌入（添加多层感知）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embed = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            #nn.SiLU(),
            nn.Linear(dim*4, dim*4),
            #nn.SiLU(),
            #nn.Linear(dim, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.embed(emb)

import math
import torch

def sinusoidal_embedding(t, dim):
    """
    Args:
        t: 时间步张量 [batch_size, ]
        dim: 嵌入维度
    Returns:
        嵌入向量 [batch_size, dim]
    """
    device = t.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t.float()[:, None] * emb[None, :]  # [batch_size, half_dim]

    # 拼接正弦和余弦分量
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # 处理奇数维度情况
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')

    return emb

#%%
class ConditionalEmbedding(nn.Module):
    def __init__(self, num_classes, time_dim=256, label_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim*4)
        )
        self.label_embed = nn.Embedding(num_classes, label_dim)
        self.fusion = nn.Sequential(
            nn.Linear(time_dim*4 + label_dim, time_dim*2),
            nn.SiLU(),
            nn.Linear(time_dim*2, time_dim)
        )

    def forward(self, t, y):
        # t: [B,] 时间步
        # y: [B,] 标签
        t_emb = sinusoidal_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)  # [B, time_dim]

        l_emb = self.label_embed(y).squeeze(1)     # [B, label_dim]

        # 融合时间与标签信息
        combined = torch.cat([t_emb, l_emb], dim=1)
        return self.fusion(combined)    # [B, time_dim]

#%% md
### Group Norm
#%%
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
#%% md
### 采样
#%%
# 上采样（反卷积）
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

# 下采样
def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)
#%% md
### Attention
#%%
from torch import einsum, softmax
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b"
                     " h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
#%%
class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)
#%% md
### Block
#%%
class ConvNextBlock(nn.Module):
    """A ConvNet for the 2020s"""

    def __init__(self, in_ch, out_ch, time_embed_dim, mult=2, norm=True):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(time_embed_dim, out_ch*2))

        self.ds_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=in_ch)

        self.net = nn.Sequential(
            nn.GroupNorm(1, in_ch) if norm else nn.Identity(),
            nn.Conv2d(in_ch, out_ch * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_ch * mult),
            nn.Conv2d(out_ch * mult, out_ch, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)
#%%
class MyBlock(nn.Module):
    """简化后的基础块（去除残差和注意力）"""
    def __init__(self, in_ch, out_ch, time_dim, mult = 1):
        super().__init__()
        # self.time_mlp = nn.Linear(time_dim, out_ch*2)
        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(time_dim, out_ch))

        self.ds_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.conv = nn.Sequential(
            nn.GroupNorm(1, out_ch),  # 原 in_ch 改为 out_ch !
            nn.Conv2d(out_ch, out_ch * mult, 3, padding=1),  # 同步修改输入通道为 out_ch
            nn.GELU(),
            nn.GroupNorm(1, out_ch * mult),
            nn.Conv2d(out_ch * mult, out_ch, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        h = self.ds_conv(x)
        condition = self.time_mlp(t_emb)
        h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.conv(h)
        return h + self.res_conv(x)
#%%
from einops import rearrange

class ConditionalBlock(nn.Module):
    """基于你原有MyBlock改造的条件版本"""
    def __init__(self, in_ch, out_ch, cond_dim, mult=1):
        """
        Args:
            cond_dim: 条件向量的维度 (time+label的融合维度)
        """
        super().__init__()

        # 修改后的条件投影层（移除偏置项）验证条件注入的有效性
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, out_ch*2, bias=False),  # 关键修改：bias=False
            nn.GELU()
        )

        # 保持原有卷积结构
        self.ds_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.conv = nn.Sequential(
            nn.GroupNorm(1, out_ch),
            nn.Conv2d(out_ch, out_ch * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_ch * mult),
            nn.Conv2d(out_ch * mult, out_ch, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond_emb):
        """输入变化：t_emb → cond_emb (融合时间+标签的条件向量)"""
        h = self.ds_conv(x)

        # 条件注入 (scale and shift)
        scale, shift = self.cond_mlp(cond_emb).chunk(2, dim=1)  # [B, 2*out_ch] → [B, out_ch], [B, out_ch]
        h = h * (1 + scale[:, :, None, None])  # 缩放
        h = h + shift[:, :, None, None]        # 偏移

        h = self.conv(h)
        return h + self.res_conv(x)  # 保持原有残差连接


def test_myblock():
    # 测试配置
    batch_size = 64
    height, width = 24, 50
    time_dim = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试案例配置
    test_cases = [
        {"in_ch": 32, "out_ch": 64, "mult": 1},
        {"in_ch": 64, "out_ch": 128, "mult": 2},
        {"in_ch": 128, "out_ch": 256, "mult": 4}
    ]

    for case in test_cases:
        print(f"\n=== 测试配置: {case} ===")

        # 初始化模块
        block = MyBlock(
            in_ch=case["in_ch"],
            out_ch=case["out_ch"],
            time_dim=time_dim,
            mult=case["mult"]
        ).to(device)

        # 生成测试输入
        x = torch.randn(batch_size, case["in_ch"], height, width).to(device)
        t_emb = torch.randn(batch_size, time_dim).to(device)

        # 测试1: 前向传播形状
        def test_shape():
            output = block(x, t_emb)
            expected_shape = (batch_size, case["out_ch"], height, width)
            assert output.shape == expected_shape, \
                f"形状错误！期望: {expected_shape}, 实际: {output.shape}"
            print("✅ 前向传播形状测试通过")

        # 测试2: 梯度流
        def test_gradient():
            block.train()
            x.requires_grad_(True)
            output = block(x, t_emb)
            loss = output.mean()
            loss.backward()

            # 检查输入梯度
            assert x.grad is not None, "输入梯度未生成"
            # 检查参数梯度
            has_grad = any(p.grad is not None for p in block.parameters())
            assert has_grad, "参数未接收梯度"
            print("✅ 梯度流测试通过")

        # 测试3: 设备兼容性
        def test_device():
            cpu_block = MyBlock(**case, time_dim=time_dim).cpu()
            cpu_x = x.cpu()
            cpu_t = t_emb.cpu()
            output = cpu_block(cpu_x, cpu_t)
            assert output.device.type == "cpu", "应生成CPU张量"
            print("✅ CPU兼容性测试通过")

            if torch.cuda.is_available():
                gpu_output = block(x, t_emb)
                assert gpu_output.is_cuda, "应生成CUDA张量"
                print("✅ GPU兼容性测试通过")

        # 执行测试
        test_shape()
        test_gradient()
        test_device()

import torch
import torch.nn as nn
from einops import rearrange

def test_conditional_myblock():
    # 测试配置
    batch_size = 4
    height, width = 24, 50
    cond_dim = 256  # 条件向量维度
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试案例
    test_cases = [
        {"in_ch": 32, "out_ch": 64, "mult": 1},
        {"in_ch": 64, "out_ch": 128, "mult": 2},
        {"in_ch": 128, "out_ch": 256, "mult": 4}
    ]

    for case in test_cases:
        print(f"\n=== 测试配置 {case} ===")

        # 初始化模块
        block = ConditionalBlock(
            in_ch=case["in_ch"],
            out_ch=case["out_ch"],
            cond_dim=cond_dim,
            mult=case["mult"]
        ).to(device)

        # 生成测试输入
        x = torch.randn(batch_size, case["in_ch"], height, width).to(device)
        cond_emb = torch.randn(batch_size, cond_dim).to(device)

        # 测试1: 前向传播形状
        def test_shape():
            output = block(x, cond_emb)
            expected_shape = (batch_size, case["out_ch"], height, width)
            assert output.shape == expected_shape, \
                f"形状错误！期望: {expected_shape}, 实际: {output.shape}"
            print("✅ 形状测试通过")

        # 测试2: 梯度流
        def test_gradient():
            block.train()
            x.requires_grad_(True)
            output = block(x, cond_emb)
            loss = output.mean()
            loss.backward()

            # 检查输入梯度
            assert x.grad is not None, "输入梯度未生成"
            # 检查条件投影层梯度
            assert block.cond_mlp[0].weight.grad is not None, "条件投影层未更新"
            print("✅ 梯度测试通过")

        # 测试3: 条件注入有效性
        def test_condition_effect():
            # 相同输入不同条件
            cond1 = torch.randn_like(cond_emb)
            cond2 = torch.randn_like(cond_emb)

            out1 = block(x, cond1)
            out2 = block(x, cond2)

            # 确保不同条件产生不同输出
            assert not torch.allclose(out1, out2, atol=1e-6), "条件未影响输出"

            # 零条件测试
            zero_cond = torch.zeros_like(cond_emb)
            out_zero = block(x, zero_cond)
            scale, shift = block.cond_mlp(zero_cond).chunk(2, dim=1)

            # 验证数学正确性：当条件为零时，h = h_conv + residual
            h_conv = block.ds_conv(x)
            expected = block.conv(h_conv) + block.res_conv(x)
            assert torch.allclose(out_zero, expected, atol=1e-6), "零条件计算错误"
            print("✅ 条件注入测试通过")

        # 测试4: 设备兼容性
        def test_device():
            cpu_block = ConditionalBlock(**case, cond_dim=cond_dim).cpu()
            cpu_x = x.cpu()
            cpu_cond = cond_emb.cpu()
            output = cpu_block(cpu_x, cpu_cond)
            assert output.device.type == "cpu", "应生成CPU张量"
            print("✅ CPU兼容性测试通过")

            if torch.cuda.is_available():
                gpu_output = block(x, cond_emb)
                assert gpu_output.is_cuda, "应生成CUDA张量"
                print("✅ GPU兼容性测试通过")

        # 执行测试
        test_shape()
        test_gradient()
        test_condition_effect()
        test_device()

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class EnhancedDiffusionUNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        chs = [1, 64, 128, 256]

        self.time_embed = EnhancedTimeEmbedding(time_dim)

        # 下采样路径
        self.down = nn.ModuleList([
            MySequential(
                MyBlock(chs[i], chs[i+1], time_dim*4),
                MyBlock(chs[i+1], chs[i+1], time_dim*4),
                #ChannelAttention(chs[i+1])
                Residual(PreNorm(chs[i+1], LinearAttention(chs[i+1])))
            ) for i in range(len(chs)-1)
        ])

        # 中间层
        self.mid = MySequential(
            MyBlock(chs[-1], chs[-1], time_dim*4),
            #ChannelAttention(chs[-1]),
            Residual(PreNorm(chs[-1], Attention(chs[-1]))),
            MyBlock(chs[-1], chs[-1], time_dim*4)
        )

        # 上采样路径
        self.up = nn.ModuleList([
            MySequential(
                MyBlock(chs[i+1]*2, chs[i], time_dim*4),
                MyBlock(chs[i], chs[i], time_dim*4),
                #ChannelAttention(chs[i])
                Residual(PreNorm(chs[i], LinearAttention(chs[i])))
            ) for i in reversed(range(len(chs)-1))
        ])

        self.final = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x, t):
        x = x.unsqueeze(1)  # [B,1,24,50]
        t_emb = self.time_embed(t)
        skips = []

        # 编码器
        for block in self.down:
            x = block(x, t_emb)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=(2,1))

        # 中间处理
        x = self.mid(x, t_emb)

        # 解码器
        for i, block in enumerate(self.up):
            x = F.interpolate(x, scale_factor=(2,1), mode='nearest')
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = block(x, t_emb)

        return self.final(x).squeeze(1)

def test_diffusion_unet():
    # 配置测试参数
    batch_size = 64
    input_shape = (24, 50)  # 你的数据维度
    timesteps = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    model = EnhancedDiffusionUNet(time_dim=128).to(device)

    # 测试案例1: 基础前向传播
    def test_forward_pass():
        # 生成模拟输入
        x = torch.randn(batch_size, *input_shape).to(device)  # [64,24,50]
        t = torch.randint(0, timesteps, (batch_size,)).to(device)  # 时间步
        # 前向传播
        output = model(x, t)

        # 验证输出形状
        assert output.shape == (batch_size, *input_shape), \
            f"输出形状错误！期望: {(batch_size, *input_shape)}, 实际: {output.shape}"

        print("✅ 前向传播测试通过")

    # 测试案例2: 设备兼容性
    def test_device_compatibility():
        cpu_model = EnhancedDiffusionUNet(time_dim=128).cpu()
        x_cpu = torch.randn(batch_size, *input_shape).cpu()
        t_cpu = torch.randint(0, timesteps, (batch_size,)).cpu()

        output_cpu = cpu_model(x_cpu, t_cpu)
        assert not output_cpu.is_cuda, "CPU模型不应产生GPU张量"

        if torch.cuda.is_available():
            gpu_output = model(x_cpu.to(device), t_cpu.to(device))
            assert gpu_output.is_cuda, "GPU模型应产生CUDA张量"

        print("✅ 设备兼容性测试通过")

    # 测试案例3: 梯度检查
    def test_gradient_flow():
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(batch_size, *input_shape).to(device)
        t = torch.randint(0, timesteps, (batch_size,)).to(device)
        target = torch.randn_like(x)

        # 模拟训练步骤
        optimizer.zero_grad()
        pred = model(x, t)
        loss = F.mse_loss(pred, target)
        loss.backward()

        # 检查梯度是否存在
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "模型参数未接收到梯度"

        print("✅ 梯度流测试通过")

    # 执行测试
    print("=== 开始Diffusion U-Net测试 ===")
    test_forward_pass()
    test_device_compatibility()
    test_gradient_flow()
    print("=== 所有测试通过 ===")

class ConditionalDiffusionUNet(nn.Module):
    def __init__(self, num_classes, time_dim=128, label_dim=64):
        super().__init__()
        chs = [1, 64, 128, 256]

        # 替换为条件嵌入层
        self.cond_embed = ConditionalEmbedding(
            num_classes=num_classes,
            time_dim=time_dim,
            label_dim=label_dim
        )
        cond_dim = time_dim  # 条件向量的总维度

        # 下采样路径（修改所有MyBlock的cond_dim）
        self.down = nn.ModuleList([
            MySequential(
                ConditionalBlock(chs[i], chs[i+1], cond_dim=cond_dim),
                ConditionalBlock(chs[i+1], chs[i+1], cond_dim=cond_dim),
                Residual(PreNorm(chs[i+1], LinearAttention(chs[i+1])))
            ) for i in range(len(chs)-1)
        ])

        # 中间层
        self.mid = MySequential(
            ConditionalBlock(chs[-1], chs[-1], cond_dim=cond_dim),
            Residual(PreNorm(chs[-1], Attention(chs[-1]))),
            ConditionalBlock(chs[-1], chs[-1], cond_dim=cond_dim)
        )

        # 上采样路径
        self.up = nn.ModuleList([
            MySequential(
                ConditionalBlock(chs[i+1]*2, chs[i], cond_dim=cond_dim),
                ConditionalBlock(chs[i], chs[i], cond_dim=cond_dim),
                Residual(PreNorm(chs[i], LinearAttention(chs[i])))
            ) for i in reversed(range(len(chs)-1))
        ])

        self.final = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x, t, y):
        """新增标签y作为输入"""
        x = x.unsqueeze(1)  # [B,1,24,50]

        cond_emb = self.cond_embed(t, y)  # 获取融合条件向量
        skips = []

        # 编码器（传递cond_emb）
        for block in self.down:
            x = block(x, cond_emb)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=(2,1))

        # 中间处理
        x = self.mid(x, cond_emb)

        # 解码器
        for i, block in enumerate(self.up):
            x = F.interpolate(x, scale_factor=(2,1), mode='nearest')
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = block(x, cond_emb)

        return self.final(x).squeeze(1)


def test_conditional_diffusion_unet():
    # 配置测试参数
    batch_size = 32
    input_shape = (24, 50)    # 输入数据维度
    num_classes = 10          # 类别数量
    timesteps = 1000          # 扩散步数
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化条件模型
    model = ConditionalDiffusionUNet(
        num_classes=num_classes,
        time_dim=128,
        label_dim=64
    ).to(device)

    # 测试案例1: 基础前向传播
    def test_forward_pass():
        # 生成模拟输入（注意标签维度）
        x = torch.randn(batch_size, *input_shape).to(device)  # [32,24,50]
        t = torch.randint(0, timesteps, (batch_size,)).to(device)
        y = torch.randint(0, num_classes, (batch_size,)).to(device)  # 关键：一维标签

        # 前向传播
        output = model(x, t, y)

        # 验证输出形状
        assert output.shape == x.shape, \
            f"形状不匹配！输入: {x.shape}, 输出: {output.shape}"
        print("✅ 前向传播测试通过")

    # 测试案例2: 设备兼容性
    def test_device_compatibility():
        # CPU测试
        cpu_model = ConditionalDiffusionUNet(num_classes=num_classes).cpu()
        x_cpu = torch.randn(batch_size, *input_shape).cpu()
        t_cpu = torch.randint(0, timesteps, (batch_size,)).cpu()
        y_cpu = torch.randint(0, num_classes, (batch_size,)).cpu()

        output_cpu = cpu_model(x_cpu, t_cpu, y_cpu)
        assert output_cpu.device.type == "cpu", "CPU模型应生成CPU张量"

        # GPU测试（如果可用）
        if torch.cuda.is_available():
            output_gpu = model(x_cpu.to(device), t_cpu.to(device), y_cpu.to(device))
            assert output_gpu.is_cuda, "GPU模型应生成CUDA张量"
        print("✅ 设备兼容性测试通过")

    # 测试案例3: 梯度流检查
    def test_gradient_flow():
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 生成模拟数据
        x = torch.randn(batch_size, *input_shape).to(device)
        t = torch.randint(0, timesteps, (batch_size,)).to(device)
        y = torch.randint(0, num_classes, (batch_size,)).to(device)
        target = torch.randn_like(x)

        # 模拟训练步骤
        optimizer.zero_grad()
        pred = model(x, t, y)
        loss = F.mse_loss(pred, target)
        loss.backward()

        # 检查关键层梯度
        assert model.cond_embed.label_embed.weight.grad is not None, "标签嵌入层无梯度"
        assert model.down[0][0].cond_mlp[0].weight.grad is not None, "条件投影层无梯度"
        assert model.final.weight.grad is not None, "输出层无梯度"
        print("✅ 梯度流测试通过")

    # 测试案例4: 条件有效性
    def test_condition_effectiveness():
        # 固定输入和时间步
        x = torch.randn(1, *input_shape).to(device)
        t = torch.randint(0, timesteps, (1,)).to(device)

        # 测试不同标签
        y1 = torch.tensor([3], device=device)
        y2 = torch.tensor([7], device=device)
        out1 = model(x, t, y1)
        out2 = model(x, t, y2)
        assert not torch.allclose(out1, out2, atol=1e-6), "不同标签应产生不同输出"

        # 测试不同时间步
        t1 = torch.tensor([200], device=device)
        t2 = torch.tensor([800], device=device)
        out3 = model(x, t1, y1)
        out4 = model(x, t2, y1)
        assert not torch.allclose(out3, out4, atol=1e-6), "不同时间步应产生不同输出"
        print("✅ 条件有效性测试通过")

    # 测试案例5: 极端输入稳定性
    def test_extreme_inputs():
        # 零输入测试
        x_zero = torch.zeros(batch_size, *input_shape).to(device)
        t = torch.randint(0, timesteps, (batch_size,)).to(device)
        y = torch.randint(0, num_classes, (batch_size,)).to(device)
        output = model(x_zero, t, y)
        assert not torch.isnan(output).any(), "零输入产生NaN"

        # 极大值输入
        x_large = 1e5 * torch.randn_like(x_zero)
        output = model(x_large, t, y)
        assert not torch.isinf(output).any(), "极大输入产生Inf"
        print("✅ 极端输入测试通过")

    # 执行所有测试
    print("\n=== 开始条件扩散U-Net测试 ===")
    test_forward_pass()
    test_device_compatibility()
    test_gradient_flow()
    test_condition_effectiveness()
    test_extreme_inputs()
    print("=== 所有测试通过 ===")

def p_sample_cond(model, x, t, t_index, y):
    """带条件标签的单步去噪采样"""
    with torch.no_grad():
        # 添加通道维度并传入标签y
        pred_noise = model(x, t, y).squeeze(1)  # [B,24,50]

    # 调整系数维度
    sqrt_recip_alphas_t = (1 / torch.sqrt(alphas[t])).view(-1, 1, 1)  # [64] -> [64, 1, 1]
    sqrt_one_minus_alphas_bar_t = extract(one_minus_alphas_bar_sqrt, t, x.shape)

    # 去噪计算
    x_recon = sqrt_recip_alphas_t * (x - pred_noise * sqrt_one_minus_alphas_bar_t)

    if t_index > 0:
        noise = torch.randn_like(x)
        sqrt_beta_t = extract(torch.sqrt(betas), t, x.shape)
        x_recon += sqrt_beta_t * noise

    return x_recon

@torch.no_grad()
def p_sample_loop_cond(model, shape, y, device='cuda'):
    """带标签的完整采样循环"""
    # 初始化噪声和标签处理
    img = torch.randn(shape, device=device)  # [B,24,50]
    y = y.to(device)  # 标签需与模型同设备

    # 反向时间步采样
    for i in reversed(range(0, num_steps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample_cond(model, img, t, i, y)

    return img

def generate_conditional_samples(
    model,
    num_samples=16,
    labels=None,  # 可指定标签 [num_samples]
    num_classes=7,  # 你的数据类别数
    device='cuda'
):
    """条件样本生成入口"""
    # 标签处理逻辑
    if labels is None:
        # 随机生成标签
        labels = torch.randint(0, num_classes, (num_samples,))
    else:
        assert len(labels) == num_samples, "标签数量需与样本数一致"

    # 输入形状 [B,24,50]
    sample_shape = (num_samples, 24, 50)

    model.eval()
    samples = p_sample_loop_cond(
        model,
        shape=sample_shape,
        y=labels,
        device=device
    )

    # 后处理
    samples = samples.cpu().numpy()
    samples = np.clip(samples, -1, 1)  # 根据你的数据归一化范围调整

    return samples, labels

def test_conditional_generation():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")

    # 初始化模型 (参数需与训练时一致)
    num_classes = 7  # 根据你的数据集调整
    model = ConditionalDiffusionUNet(
        num_classes=num_classes,
        time_dim=128,
        label_dim=64
    ).to(device)

    # 打印模型结构
    print("\n模型结构:")
    print(model)

    # 测试用例1: 随机生成不同类别的样本
    print("\n测试用例1: 随机生成样本")
    samples, labels = generate_conditional_samples(
        model,
        num_samples=4,  # 生成4个样本
        num_classes=num_classes,
        device=device
    )

    # 验证输出形状
    assert samples.shape == (4, 24, 50), f"样本形状错误: 期望 (4,24,50)，实际 {samples.shape}"
    assert labels.shape == (4,), f"标签形状错误: 期望 (4,)，实际 {labels.shape}"
    print(f"生成样本形状: {samples.shape} | 标签形状: {labels.shape}")

    # 验证数据范围
    assert samples.min() >= -1 and samples.max() <= 1, "样本数据范围超出 [-1, 1]"
    print(f"数据范围验证通过: min={samples.min():.2f}, max={samples.max():.2f}")

    # 测试用例2: 指定特定标签生成
    print("\n测试用例2: 指定标签生成")
    target_labels = torch.tensor([0, 1, 2, 3])  # 生成4个不同类别的样本
    samples, labels = generate_conditional_samples(
        model,
        num_samples=4,
        labels=target_labels,
        device=device
    )

    # 验证标签匹配
    assert (labels == target_labels).all(), "生成标签与指定标签不匹配"
    print(f"标签匹配验证通过: {labels.tolist()}")

    # 可视化第一个样本
    plt.figure(figsize=(10, 6))
    plt.imshow(samples[0], cmap='viridis', aspect='auto')
    plt.title(f"生成样本示例 (类别={labels[0]})")
    plt.colorbar()
    plt.show()

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

def q_sample(x0, t, noise):
    """前向扩散过程：根据时间步t给x0加噪"""
    sqrt_alpha_prod = torch.sqrt(alphas_prod[t]).view(-1, 1, 1)
    sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_prod[t]).view(-1, 1, 1)
    return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

def train_conditional_diffusion():
    # 超参数配置
    config = {
        "batch_size": 64,
        "lr": 2e-4,
        "epochs": 1000,
        "num_samples": 8,        # 验证时生成的样本数
        "save_interval": 100,     # 保存间隔（epoch）
        "grad_clip": 1.0,         # 梯度裁剪阈值
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # 准备数据集
    dataset = TensorDataset(x0, labels)  # x0: [N,24,50], labels: [N]
    dataloader = DataLoader(dataset,
                          batch_size=config["batch_size"],
                          shuffle=True,
                          pin_memory=True)

    # 初始化模型
    model = ConditionalDiffusionUNet(
        num_classes=num_classes,
        time_dim=128,
        label_dim=64
    ).to(config["device"])

    # 优化器与学习率调度
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, config["epochs"]+1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_x0, batch_labels in pbar:
            # 数据准备
            batch_x0 = batch_x0.to(config["device"])  # [B,24,50]
            batch_labels = batch_labels.to(config["device"])  # [B]

            # 随机采样时间步
            b = batch_x0.size(0)
            t = torch.randint(0, num_steps, (b,), device=config["device"])

            # 生成噪声
            noise = torch.randn_like(batch_x0)

            # 前向扩散
            xt = q_sample(batch_x0, t, noise)

            # 模型预测
            pred_noise = model(xt, t, batch_labels)

            # 计算损失
            loss = F.mse_loss(pred_noise, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            # 记录损失
            epoch_loss += loss.item() * b
            pbar.set_postfix({"loss": loss.item()})

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch} | Loss: {epoch_loss:.4f}")

        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_conditional_diffusion.pth")

        # 定期采样验证
        if epoch % config["save_interval"] == 0:
            model.eval()
            with torch.no_grad():
                # 生成每个类别的样本
                for label in range(num_classes):
                    samples = generate_class_samples(
                        model=model,
                        num_samples=config["num_samples"],
                        label=label,
                        device=config["device"]
                    )
                    save_samples(samples, f"epoch{epoch}_class{label}.npy")

            # 保存检查点
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": epoch_loss
            }, f"checkpoint_epoch{epoch}.pth")

def generate_class_samples(model, num_samples, label, device):
    """生成指定类别的样本"""
    model.eval()
    labels = torch.full((num_samples,), label, device=device)
    shape = (num_samples, 24, 50)

    # 初始噪声
    xt = torch.randn(shape, device=device)

    # 迭代去噪
    for t in reversed(range(num_steps)):
        timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
        xt = p_sample(model, xt, timesteps, labels)

    # 后处理
    samples = xt.cpu().numpy()
    samples = (samples - samples.min()) / (samples.max() - samples.min())  # 归一化到[0,1]
    return samples

def p_sample(model, xt, t, y):
    """单步去噪采样"""
    alpha_t = extract(alphas, t, xt.shape)
    beta_t = extract(betas, t, xt.shape)

    with torch.no_grad():
        pred_noise = model(xt, t, y)

    # 计算去噪结果
    noise = torch.randn_like(xt) if t[0] > 0 else 0
    xt_prev = 1 / torch.sqrt(alpha_t) * (
        xt - (beta_t / torch.sqrt(1 - alphas_prod[t])) * pred_noise
    ) + torch.sqrt(beta_t) * noise

    return xt_prev

def save_samples(samples, filename):
    """保存样本到文件"""
    np.save(filename, samples)
    print(f"Saved samples to {filename}")
