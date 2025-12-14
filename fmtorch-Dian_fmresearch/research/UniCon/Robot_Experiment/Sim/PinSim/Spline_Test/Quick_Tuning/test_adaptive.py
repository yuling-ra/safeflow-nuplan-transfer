#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本：验证自适应密度功能
测试不同的幂函数参数配置
"""
import numpy as np
import matplotlib.pyplot as plt

def power_function_density(t, center, width, height, decay_rate, power):
    """幂函数密度boost"""
    normalized_dist = (t - center) / width
    boost = height * np.exp(-decay_rate * np.abs(normalized_dist) ** power)
    return boost


def test_power_function_shapes():
    """测试不同参数下的幂函数形状"""
    t = np.linspace(0, 1, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Test 1: Different centers
    ax1 = axes[0, 0]
    for center in [0.2, 0.5, 0.8]:
        boost = power_function_density(t, center, 0.2, 1.0, 3.0, 2)
        ax1.plot(t, boost, linewidth=2, label=f'center={center}')
    ax1.set_title('不同 center 值的效果', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('Density Boost')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test 2: Different widths
    ax2 = axes[0, 1]
    for width in [0.1, 0.3, 0.5]:
        boost = power_function_density(t, 0.5, width, 1.0, 3.0, 2)
        ax2.plot(t, boost, linewidth=2, label=f'width={width}')
    ax2.set_title('不同 width 值的效果', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Density Boost')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Test 3: Different heights
    ax3 = axes[1, 0]
    for height in [0.5, 1.0, 2.0]:
        boost = power_function_density(t, 0.5, 0.3, height, 3.0, 2)
        ax3.plot(t, boost, linewidth=2, label=f'height={height}')
    ax3.set_title('不同 height 值的效果', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Normalized Time')
    ax3.set_ylabel('Density Boost')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Test 4: Different decay rates and powers
    ax4 = axes[1, 1]
    for decay, power in [(2.0, 2), (5.0, 2), (3.0, 4)]:
        boost = power_function_density(t, 0.5, 0.3, 1.0, decay, power)
        ax4.plot(t, boost, linewidth=2, label=f'decay={decay}, power={power}')
    ax4.set_title('不同 decay_rate 和 power 的效果', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Normalized Time')
    ax4.set_ylabel('Density Boost')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('幂函数参数效果测试', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('power_function_test.png', dpi=150, bbox_inches='tight')
    print("✅ 已生成: power_function_test.png")
    plt.close()


def test_control_point_calculation():
    """测试控制点数量计算"""
    T = 1000
    M_base = 50
    
    configs = [
        {'center': 0.5, 'width': 0.3, 'height': 0.0, 'decay_rate': 3.0, 'power': 2, 'name': '禁用 (height=0)'},
        {'center': 0.5, 'width': 0.3, 'height': 1.0, 'decay_rate': 3.0, 'power': 2, 'name': '中等增强'},
        {'center': 0.5, 'width': 0.3, 'height': 2.0, 'decay_rate': 3.0, 'power': 2, 'name': '高增强'},
        {'center': 0.3, 'width': 0.2, 'height': 1.5, 'decay_rate': 5.0, 'power': 2, 'name': '窄峰前段'},
    ]
    
    print("\n" + "="*70)
    print("控制点数量计算测试")
    print("="*70)
    print(f"基线控制点数: {M_base}")
    print(f"时间步数: {T}")
    print()
    
    for config in configs:
        t = np.linspace(0, 1, T)
        density_curve = power_function_density(
            t, config['center'], config['width'], 
            config['height'], config['decay_rate'], config['power']
        )
        mean_boost = np.mean(density_curve)
        M_adaptive = int(M_base * (1 + mean_boost))
        M_adaptive = max(M_adaptive, M_base)
        
        print(f"配置: {config['name']}")
        print(f"  参数: center={config['center']}, width={config['width']}, "
              f"height={config['height']}, decay={config['decay_rate']}, power={config['power']}")
        print(f"  平均boost: {mean_boost:.4f}")
        print(f"  自适应控制点: {M_adaptive} (增加 +{M_adaptive - M_base})")
        print(f"  增加百分比: {(M_adaptive - M_base) / M_base * 100:.1f}%")
        print()


if __name__ == '__main__':
    print("="*70)
    print("自适应密度功能测试")
    print("="*70)
    
    # Test 1: 幂函数形状
    print("\n[测试1] 生成幂函数形状对比图...")
    test_power_function_shapes()
    
    # Test 2: 控制点计算
    print("\n[测试2] 测试控制点数量计算...")
    test_control_point_calculation()
    
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)

