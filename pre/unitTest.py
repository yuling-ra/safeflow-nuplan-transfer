import unittest
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

# 复制您的基础接口
class Sampleable(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

class ConditionalProbabilityPath(ABC):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        self.p_simple = p_simple
        self.p_data = p_data
    
    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        num_samples = t.shape[0]
        print(f"[DEBUG] sample_marginal_path - 输入时间t形状: {t.shape}")
        
        z, y = self.sample_conditioning_variable(num_samples)
        print(f"[DEBUG] sample_marginal_path - 条件变量z形状: {z.shape}")
        print(f"[DEBUG] sample_marginal_path - 标签y形状: {y.shape if y is not None else 'None'}")
        
        x = self.sample_conditional_path(z, t)
        print(f"[DEBUG] sample_marginal_path - 输出轨迹x形状: {x.shape}")
        
        return x
    
    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    @abstractmethod  
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

# 测试用的具体实现类
class TestTrajectoryDataset(Sampleable):
    """测试用的轨迹数据集"""
    def __init__(self, seq_len=50, dim=2, num_classes=3):
        self.seq_len = seq_len
        self.dim = dim
        self.num_classes = num_classes
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        print(f"[DEBUG] TestTrajectoryDataset.sample - 请求样本数: {num_samples}")
        
        # 生成简单的轨迹数据
        trajectories = torch.randn(num_samples, self.seq_len, self.dim)
        labels = torch.randint(0, self.num_classes, (num_samples,))
        
        print(f"[DEBUG] TestTrajectoryDataset.sample - 生成轨迹形状: {trajectories.shape}")
        print(f"[DEBUG] TestTrajectoryDataset.sample - 生成标签形状: {labels.shape}")
        print(f"[DEBUG] TestTrajectoryDataset.sample - 轨迹数据范围: [{trajectories.min():.3f}, {trajectories.max():.3f}]")
        print(f"[DEBUG] TestTrajectoryDataset.sample - 标签范围: [{labels.min()}, {labels.max()}]")
        
        return trajectories, labels

class TestNoiseDataset(Sampleable):
    """测试用的噪声数据集"""
    def __init__(self, seq_len=50, dim=2):
        self.seq_len = seq_len
        self.dim = dim
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        print(f"[DEBUG] TestNoiseDataset.sample - 请求噪声样本数: {num_samples}")
        
        noise = torch.randn(num_samples, self.seq_len, self.dim)
        
        print(f"[DEBUG] TestNoiseDataset.sample - 生成噪声形状: {noise.shape}")
        print(f"[DEBUG] TestNoiseDataset.sample - 噪声数据范围: [{noise.min():.3f}, {noise.max():.3f}]")
        
        return noise, None

class TestConditionalProbabilityPath(ConditionalProbabilityPath):
    """测试用的条件概率路径"""
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__(p_simple, p_data)
    
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        print(f"[DEBUG] sample_conditioning_variable - 请求条件变量数量: {num_samples}")
        
        z, y = self.p_data.sample(num_samples)
        
        print(f"[DEBUG] sample_conditioning_variable - 返回条件变量z形状: {z.shape}")
        print(f"[DEBUG] sample_conditioning_variable - 返回标签y形状: {y.shape}")
        
        return z, y
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        print(f"[DEBUG] sample_conditional_path - 输入条件变量z形状: {z.shape}")
        print(f"[DEBUG] sample_conditional_path - 输入时间t形状: {t.shape}")
        
        num_samples = z.shape[0]
        noise, _ = self.p_simple.sample(num_samples)
        
        # 简单的线性插值：x_t = (1-t) * z + t * noise
        t_expanded = t.view(-1, 1, 1)  # (batch, 1, 1)
        x_t = (1 - t_expanded) * z + t_expanded * noise
        
        print(f"[DEBUG] sample_conditional_path - 时间t扩展后形状: {t_expanded.shape}")
        print(f"[DEBUG] sample_conditional_path - 输出轨迹x_t形状: {x_t.shape}")
        print(f"[DEBUG] sample_conditional_path - 输出数据范围: [{x_t.min():.3f}, {x_t.max():.3f}]")
        
        return x_t

# Unittest 测试类
class TestTrajectoryInterfaces(unittest.TestCase):
    """轨迹生成接口的单元测试"""
    
    def setUp(self):
        """测试前的初始化"""
        print("\n" + "="*60)
        print(f"开始测试: {self._testMethodName}")
        print("="*60)
        
        self.seq_len = 50
        self.dim = 2
        self.num_classes = 3
        self.batch_size = 4
        
        # 创建测试实例
        self.trajectory_data = TestTrajectoryDataset(self.seq_len, self.dim, self.num_classes)
        self.noise_data = TestNoiseDataset(self.seq_len, self.dim)
        self.cond_path = TestConditionalProbabilityPath(self.noise_data, self.trajectory_data)
    
    def test_sampleable_trajectory_dataset(self):
        """测试轨迹数据集的采样接口"""
        print("\n[TEST] 测试轨迹数据集采样...")
        
        trajectories, labels = self.trajectory_data.sample(self.batch_size)
        
        # 验证形状
        expected_traj_shape = (self.batch_size, self.seq_len, self.dim)
        expected_label_shape = (self.batch_size,)
        
        self.assertEqual(trajectories.shape, expected_traj_shape, 
                        f"轨迹形状错误: 期望{expected_traj_shape}, 实际{trajectories.shape}")
        self.assertEqual(labels.shape, expected_label_shape,
                        f"标签形状错误: 期望{expected_label_shape}, 实际{labels.shape}")
        
        # 验证数据类型
        self.assertEqual(trajectories.dtype, torch.float32, "轨迹数据类型应为float32")
        self.assertEqual(labels.dtype, torch.int64, "标签数据类型应为int64")
        
        # 验证标签范围
        self.assertTrue(torch.all(labels >= 0), "标签应该非负")
        self.assertTrue(torch.all(labels < self.num_classes), f"标签应该小于{self.num_classes}")
        
        print("[PASS] 轨迹数据集测试通过!")
    
    def test_sampleable_noise_dataset(self):
        """测试噪声数据集的采样接口"""
        print("\n[TEST] 测试噪声数据集采样...")
        
        noise, labels = self.noise_data.sample(self.batch_size)
        
        # 验证形状
        expected_noise_shape = (self.batch_size, self.seq_len, self.dim)
        
        self.assertEqual(noise.shape, expected_noise_shape,
                        f"噪声形状错误: 期望{expected_noise_shape}, 实际{noise.shape}")
        self.assertIsNone(labels, "噪声数据集不应返回标签")
        
        # 验证数据类型和分布
        self.assertEqual(noise.dtype, torch.float32, "噪声数据类型应为float32")
        
        # 简单的高斯分布检验
        noise_mean = noise.mean().item()
        noise_std = noise.std().item()
        self.assertAlmostEqual(noise_mean, 0.0, delta=0.3, msg="噪声均值应接近0")
        self.assertAlmostEqual(noise_std, 1.0, delta=0.3, msg="噪声标准差应接近1")
        
        print("[PASS] 噪声数据集测试通过!")
    
    def test_conditional_probability_path_sample_conditioning_variable(self):
        """测试条件概率路径的条件变量采样"""
        print("\n[TEST] 测试条件变量采样...")
        
        z, y = self.cond_path.sample_conditioning_variable(self.batch_size)
        
        # 验证形状
        expected_z_shape = (self.batch_size, self.seq_len, self.dim)
        expected_y_shape = (self.batch_size,)
        
        self.assertEqual(z.shape, expected_z_shape,
                        f"条件变量z形状错误: 期望{expected_z_shape}, 实际{z.shape}")
        self.assertEqual(y.shape, expected_y_shape,
                        f"标签y形状错误: 期望{expected_y_shape}, 实际{y.shape}")
        
        print("[PASS] 条件变量采样测试通过!")
    
    def test_conditional_probability_path_sample_conditional_path(self):
        """测试条件概率路径的条件路径采样"""
        print("\n[TEST] 测试条件路径采样...")
        
        # 准备输入
        z, y = self.cond_path.sample_conditioning_variable(self.batch_size)
        t = torch.rand(self.batch_size, 1, 1)  # 随机时间步
        
        x_t = self.cond_path.sample_conditional_path(z, t)
        
        # 验证形状
        expected_shape = (self.batch_size, self.seq_len, self.dim)
        self.assertEqual(x_t.shape, expected_shape,
                        f"条件路径输出形状错误: 期望{expected_shape}, 实际{x_t.shape}")
        
        # 验证边界条件
        # 当t=0时，应该接近z
        t_zero = torch.zeros(self.batch_size, 1, 1)
        x_t_zero = self.cond_path.sample_conditional_path(z, t_zero)
        diff_zero = torch.abs(x_t_zero - z).mean()
        self.assertLess(diff_zero.item(), 0.1, "t=0时输出应接近原始轨迹")
        
        print("[PASS] 条件路径采样测试通过!")
    
    def test_sample_marginal_path_integration(self):
        """测试边际路径采样的完整流程"""
        print("\n[TEST] 测试边际路径采样完整流程...")
        
        # 准备时间输入
        t = torch.rand(self.batch_size, 1, 1)
        
        x = self.cond_path.sample_marginal_path(t)
        
        # 验证输出形状
        expected_shape = (self.batch_size, self.seq_len, self.dim)
        self.assertEqual(x.shape, expected_shape,
                        f"边际路径输出形状错误: 期望{expected_shape}, 实际{x.shape}")
        
        # 验证数据有效性
        self.assertFalse(torch.isnan(x).any(), "输出不应包含NaN")
        self.assertFalse(torch.isinf(x).any(), "输出不应包含Inf")
        
        print("[PASS] 边际路径采样完整流程测试通过!")
    
    def test_different_batch_sizes(self):
        """测试不同批次大小的处理"""
        print("\n[TEST] 测试不同批次大小...")
        
        for batch_size in [1, 3, 8, 16]:
            print(f"\n  测试批次大小: {batch_size}")
            
            t = torch.rand(batch_size, 1, 1)
            x = self.cond_path.sample_marginal_path(t)
            
            expected_shape = (batch_size, self.seq_len, self.dim)
            self.assertEqual(x.shape, expected_shape,
                            f"批次大小{batch_size}时形状错误: 期望{expected_shape}, 实际{x.shape}")
        
        print("[PASS] 不同批次大小测试通过!")
    
    def test_time_tensor_formats(self):
        """测试不同时间张量格式的处理"""
        print("\n[TEST] 测试不同时间张量格式...")
        
        formats = [
            ((self.batch_size, 1, 1), "3D格式"),
            ((self.batch_size, 1, 1, 1), "4D格式"),
        ]
        
        for shape, desc in formats:
            print(f"\n  测试时间张量格式: {desc} - {shape}")
            
            t = torch.rand(*shape)
            
            try:
                x = self.cond_path.sample_marginal_path(t)
                expected_shape = (self.batch_size, self.seq_len, self.dim)
                self.assertEqual(x.shape, expected_shape,
                                f"时间格式{desc}时输出形状错误")
                print(f"    [PASS] {desc}处理成功")
            except Exception as e:
                self.fail(f"时间格式{desc}处理失败: {e}")
        
        print("[PASS] 不同时间张量格式测试通过!")

# 运行测试的主函数
def run_debug_tests():
    """运行所有调试测试"""
    print("开始运行轨迹生成模型接口调试测试...")
    print("这将验证所有tensor形状和数据流的正确性")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加所有测试
    test_methods = [
        'test_sampleable_trajectory_dataset',
        'test_sampleable_noise_dataset',
        'test_conditional_probability_path_sample_conditioning_variable',
        'test_conditional_probability_path_sample_conditional_path',
        'test_sample_marginal_path_integration',
        'test_different_batch_sizes',
        'test_time_tensor_formats'
    ]
    
    for method in test_methods:
        test_suite.addTest(TestTrajectoryInterfaces(method))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 所有测试通过! 您的接口tensor形状处理正确!")
    else:
        print("\n❌ 存在测试失败，请检查上述错误信息")
    
    print("="*60)
    
    return result

if __name__ == "__main__":
    # 运行调试测试
    run_debug_tests()