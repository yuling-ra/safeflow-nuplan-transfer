#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统资源监控模块
监控 CPU、内存使用，防止系统过载
"""
import psutil
import time
import os
from typing import Tuple


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, 
                 cpu_threshold: float = 85.0,
                 memory_threshold: float = 85.0,
                 check_interval: float = 1.0):
        """
        参数:
            cpu_threshold: CPU 使用率阈值 (%)
            memory_threshold: 内存使用率阈值 (%)
            check_interval: 检查间隔 (秒)
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.last_check = 0
        
    def check_resources(self) -> Tuple[bool, str]:
        """
        检查系统资源
        返回: (是否安全, 警告信息)
        """
        current_time = time.time()
        
        # 避免频繁检查
        if current_time - self.last_check < self.check_interval:
            return True, ""
        
        self.last_check = current_time
        
        # 检查 CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > self.cpu_threshold:
            msg = f"CPU usage too high: {cpu_percent:.1f}% > {self.cpu_threshold}%"
            return False, msg
        
        # 检查内存
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_threshold:
            msg = f"Memory usage too high: {memory.percent:.1f}% > {self.memory_threshold}%"
            return False, msg
        
        return True, ""
    
    def get_safe_worker_count(self, requested_workers: int) -> int:
        """
        根据当前系统负载动态调整 worker 数量
        """
        cpu_count = os.cpu_count() or 4
        
        # 获取当前系统负载
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        
        # 计算可用资源
        cpu_available = (100 - cpu_percent) / 100.0
        memory_available = (100 - memory.percent) / 100.0
        
        # 根据最紧张的资源调整
        resource_factor = min(cpu_available, memory_available)
        
        # 至少保留 20% 资源给系统
        safe_factor = max(0.1, min(0.8, resource_factor))
        safe_workers = int(cpu_count * safe_factor)
        
        # 不超过请求的数量
        final_workers = min(requested_workers, safe_workers, cpu_count - 2)
        final_workers = max(1, final_workers)  # 至少1个
        
        if final_workers < requested_workers:
            print(f"[Resource Monitor] Adjusted workers: {requested_workers} -> {final_workers}")
            print(f"  CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")
        
        return final_workers
    
    def wait_for_resources(self, timeout: float = 60.0) -> bool:
        """
        等待系统资源恢复到安全水平
        返回: 是否成功等待到资源可用
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            safe, msg = self.check_resources()
            if safe:
                return True
            
            print(f"[Resource Monitor] Waiting for resources... {msg}")
            time.sleep(2.0)
        
        return False


def get_system_info():
    """获取系统信息"""
    cpu_count = os.cpu_count() or 4
    memory = psutil.virtual_memory()
    
    info = {
        'cpu_count': cpu_count,
        'cpu_percent': psutil.cpu_percent(interval=1.0),
        'memory_total_gb': memory.total / (1024**3),
        'memory_available_gb': memory.available / (1024**3),
        'memory_percent': memory.percent,
    }
    
    return info


def print_system_info():
    """打印系统信息"""
    info = get_system_info()
    print("="*70)
    print("System Information")
    print("="*70)
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"CPU Usage: {info['cpu_percent']:.1f}%")
    print(f"Memory Total: {info['memory_total_gb']:.1f} GB")
    print(f"Memory Available: {info['memory_available_gb']:.1f} GB")
    print(f"Memory Usage: {info['memory_percent']:.1f}%")
    print("="*70)


if __name__ == '__main__':
    # 测试
    print_system_info()
    
    monitor = ResourceMonitor(cpu_threshold=80.0, memory_threshold=80.0)
    safe, msg = monitor.check_resources()
    
    if safe:
        print("✓ System resources are safe")
    else:
        print(f"✗ Resource warning: {msg}")
    
    # 测试动态调整
    safe_workers = monitor.get_safe_worker_count(18)
    print(f"\nRecommended workers: {safe_workers}/18")

