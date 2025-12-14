# %% [code] cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ===================== 数据集 =====================
class TrajectoryDataset(Dataset):
    """生成三种类型的轨迹时间序列"""
    def __init__(self, num_samples=1000, seq_len=50):
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        # 预生成所有数据
        self.trajectories = []
        self.labels = []
        
        samples_per_class = num_samples // 3
        t = np.linspace(0, 2*np.pi, seq_len)
        
        for class_id in range(3):
            for _ in range(samples_per_class):
                if class_id == 0:  # 圆形
                    x = np.cos(t) * 0.5
                    y = np.sin(t) * 0.5
                elif class_id == 1:  # 直线
                    x = np.linspace(-0.5, 0.5, seq_len)
                    y = x * 0.5
                else:  # 椭圆
                    x = np.cos(t) * 0.7
                    y = np.sin(t) * 0.3
                
                traj = np.stack([x, y], axis=-1)
                # 添加小噪声增加多样性
                traj += np.random.randn(*traj.shape) * 0.02
                
                self.trajectories.append(traj)
                self.labels.append(class_id)
        
        self.trajectories = torch.tensor(self.trajectories, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]

# ===================== 时间编码器 =====================
class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: (batch_size, 1) 或 (batch_size,) 时间步 [0, 1]
        Returns:
            (batch_size, dim) 时间编码
        """
        # 确保 t 是 1D tensor
        if t.dim() == 2:
            t = t.squeeze(-1)  # (batch_size, 1) -> (batch_size,)
        
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]  # (batch, half_dim)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

# ===================== U-Net 模型 =====================
class TrajectoryUNet(nn.Module):
    """Conv1d U-Net 用于轨迹时间序列"""
    def __init__(self, 
                 seq_len=50,
                 input_dim=2,
                 hidden_dims=[64, 128, 256],
                 t_embed_dim=128,
                 num_classes=3):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU()
        )
        
        self.label_embed = nn.Embedding(num_classes, t_embed_dim)
        
        # 输入投影: (seq_len, 2) -> (seq_len, hidden_dims[0])
        self.input_proj = nn.Conv1d(input_dim, hidden_dims[0], 1)
        
        # 编码器
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i], hidden_dims[i], 3, padding=1),
                    nn.GroupNorm(8, hidden_dims[i]),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dims[i], hidden_dims[i], 3, padding=1),
                    nn.GroupNorm(8, hidden_dims[i]),
                    nn.SiLU()
                )
            )
            self.encoder_downsamples.append(
                nn.Conv1d(hidden_dims[i], hidden_dims[i+1], 3, stride=2, padding=1)
            )
        
        # 中间层
        self.mid_block = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.GroupNorm(8, hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.GroupNorm(8, hidden_dims[-1]),
            nn.SiLU()
        )
        
        # 解码器
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(hidden_dims[i], hidden_dims[i-1], 3, padding=1)
                )
            )
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i-1], hidden_dims[i-1], 3, padding=1),
                    nn.GroupNorm(8, hidden_dims[i-1]),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dims[i-1], hidden_dims[i-1], 3, padding=1),
                    nn.GroupNorm(8, hidden_dims[i-1]),
                    nn.SiLU()
                )
            )
        
        # 输出投影
        self.output_proj = nn.Conv1d(hidden_dims[0], input_dim, 1)
        
        # 时间条件投影层
        self.time_projs = nn.ModuleList([
            nn.Linear(t_embed_dim, dim) for dim in hidden_dims
        ])
    
    def forward(self, x, t, y):
        """
        Args:
            x: (batch, seq_len, 2) 输入轨迹
            t: (batch, 1) 时间步
            y: (batch,) 类别标签
        Returns:
            (batch, seq_len, 2) 预测的速度场
        """
        # 编码时间和类别
        t_emb = self.time_embed(t) + self.label_embed(y)  # (batch, t_embed_dim)
        
        # 转换为 Conv1d 格式: (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, 2, seq_len)
        x = self.input_proj(x)  # (batch, hidden_dims[0], seq_len)
        
        # 编码器
        skip_connections = []
        for i, (block, downsample) in enumerate(zip(self.encoder_blocks, self.encoder_downsamples)):
            # 添加时间条件
            t_proj = self.time_projs[i](t_emb)[:, :, None]  # (batch, dim, 1)
            x = x + t_proj
            
            x = block(x)
            skip_connections.append(x)
            x = downsample(x)
        
        # 中间层
        t_proj = self.time_projs[-1](t_emb)[:, :, None]
        x = x + t_proj
        x = self.mid_block(x)
        
        # 解码器
        for i, (upsample, block) in enumerate(zip(self.decoder_upsamples, self.decoder_blocks)):
            x = upsample(x)
            skip = skip_connections.pop()
            # 处理尺寸不匹配
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = x + skip
            x = block(x)
        
        # 输出
        x = self.output_proj(x)
        x = x.transpose(1, 2)  # (batch, seq_len, 2)
        
        return x

# ===================== 训练函数 =====================
def train_flow_matching(model, dataloader, num_epochs=500, lr=1e-3):
    """Flow Matching 训练"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    losses = []
    pbar = tqdm(range(num_epochs), desc="训练进度")
    
    for epoch in pbar:
        epoch_loss = 0
        for z, y in dataloader:
            z, y = z.to(device), y.to(device)
            batch_size = z.shape[0]
            
            # 1. 采样时间 t ~ U(0.001, 0.999) 避开端点
            t = torch.rand(batch_size, 1, device=device) * 0.998 + 0.001
            
            # 2. 采样高斯噪声
            noise = torch.randn_like(z)
            
            # 3. 线性插值路径: x_t = t·z + (1-t)·noise
            t_expanded = t[:, :, None]  # (batch, 1, 1)
            x_t = t_expanded * z + (1 - t_expanded) * noise
            
            # 4. 目标速度场（Flow Matching 理论的闭式解）
            target_v = z - noise
            
            # 5. 预测速度场
            pred_v = model(x_t, t, y)
            
            # 6. 损失函数
            loss = F.mse_loss(pred_v, target_v)
            
            # 7. 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        scheduler.step()
        
        pbar.set_postfix({'loss': f'{epoch_loss:.6f}'})
        
        if (epoch + 1) % 100 == 0:
            print(f'\nEpoch {epoch+1}, Loss: {epoch_loss:.6f}')
    
    return losses

# ===================== 采样函数 =====================
@torch.no_grad()
def sample_trajectories(model, labels, num_steps=50):
    """使用 Euler 方法求解概率流 ODE"""
    model.eval()
    
    batch_size = len(labels)
    labels = labels.to(device)
    
    # 从高斯噪声开始
    x = torch.randn(batch_size, 50, 2, device=device)
    
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t_val = i / num_steps
        t = torch.full((batch_size, 1), t_val, device=device)
        
        # 预测速度场
        v = model(x, t, labels)
        
        # Euler 积分: dx/dt = v(x, t)
        x = x + v * dt
    
    return x.cpu()

# ===================== 可视化函数 =====================
def plot_trajectories(trajectories, labels, title="轨迹"):
    """可视化轨迹"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    class_names = ['圆形', '直线', '椭圆']
    colors = ['red', 'blue', 'green']
    
    # 总览
    for class_idx in range(3):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_trajs = trajectories[mask]
            for traj in class_trajs:
                axes[0].plot(traj[:, 0], traj[:, 1], 
                           color=colors[class_idx], alpha=0.5, linewidth=1.5)
    axes[0].set_title(f'{title} - 总览')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-0.6, 0.6)
    
    # 分类别显示
    for class_idx in range(3):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_trajs = trajectories[mask]
            for traj in class_trajs:
                axes[class_idx + 1].plot(traj[:, 0], traj[:, 1], 
                                        color=colors[class_idx], alpha=0.7, linewidth=2)
        axes[class_idx + 1].set_title(f'{class_names[class_idx]} (数量: {mask.sum()})')
        axes[class_idx + 1].set_aspect('equal')
        axes[class_idx + 1].grid(True, alpha=0.3)
        axes[class_idx + 1].set_xlim(-1, 1)
        axes[class_idx + 1].set_ylim(-0.6, 0.6)
    
    plt.tight_layout()
    plt.show()

# ===================== 主程序 =====================
def main():
    print("=" * 60)
    print("Flow Matching 轨迹生成模型")
    print("=" * 60)
    
    # 创建数据集
    dataset = TrajectoryDataset(num_samples=3000, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # 可视化真实数据
    print("\n展示真实轨迹...")
    sample_indices = torch.randperm(len(dataset))[:12]
    real_trajs = dataset.trajectories[sample_indices]
    real_labels = dataset.labels[sample_indices]
    plot_trajectories(real_trajs, real_labels, "真实轨迹")
    
    # 创建模型
    model = TrajectoryUNet(
        seq_len=50,
        input_dim=2,
        hidden_dims=[64, 128, 256],
        t_embed_dim=128,
        num_classes=3
    ).to(device)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\n开始训练...")
    losses = train_flow_matching(model, dataloader, num_epochs=500, lr=1e-3)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 生成轨迹
    print("\n生成新轨迹...")
    gen_labels = torch.arange(3).repeat(4)  # 每类生成4个
    gen_trajs = sample_trajectories(model, gen_labels, num_steps=100)
    
    # 可视化生成结果
    plot_trajectories(gen_trajs, gen_labels, "生成轨迹")
    
    # 对比真实和生成
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    colors = ['red', 'blue', 'green']
    
    for class_idx in range(3):
        # 真实
        mask = real_labels == class_idx
        for traj in real_trajs[mask]:
            axes[0].plot(traj[:, 0], traj[:, 1], color=colors[class_idx], alpha=0.6, linewidth=2)
        
        # 生成
        mask = gen_labels == class_idx
        for traj in gen_trajs[mask]:
            axes[1].plot(traj[:, 0], traj[:, 1], color=colors[class_idx], alpha=0.6, linewidth=2)
    
    axes[0].set_title('真实轨迹')
    axes[1].set_title('生成轨迹')
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.6, 0.6)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

# %% [code] cell 2


# %% [code] cell 3

if __name__ == "__main__":
    main()
