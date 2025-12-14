#!/usr/bin/env bash
set -e

# ============================================================
# 完整数据集压缩基准测试 - 自动化流程
# ============================================================

OUTDIR="complete_sweep"
DATA="test_complete.npz"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${OUTDIR}/sweep_${TIMESTAMP}.log"

# 创建输出目录
mkdir -p "${OUTDIR}"

# 日志函数
log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "${LOGFILE}"
}

log "=========================================="
log "开始完整数据集压缩基准测试"
log "输出目录: ${OUTDIR}"
log "数据文件: ${DATA}"
log "=========================================="

# ============================================================
# 阶段 1: PCA 大扫描
# K=32,48,64,96 × r=8,12,16,24,32,48,64,96 → 32 组
# ============================================================

log ""
log "阶段 1/3: PCA 大扫描 (32 组配置)"
log "K = [32, 48, 64, 96]"
log "r = [8, 12, 16, 24, 32, 48, 64, 96]"
log ""

START_TIME=$(date +%s)

python3 compress_benchmark.py \
  --data "${DATA}" \
  --out "${OUTDIR}/stage1_pca_sweep" \
  --methods PCA \
  --K 32 48 64 96 \
  --latent 8 12 16 24 32 48 64 96 \
  --wq 1.0 --wdq 0.25 --wtau 0.1 \
  --seed 42 \
  2>&1 | tee -a "${LOGFILE}"

STAGE1_TIME=$(($(date +%s) - START_TIME))
log "阶段 1 完成，耗时: $((STAGE1_TIME / 60)) 分 $((STAGE1_TIME % 60)) 秒"

# ============================================================
# 阶段 2: VAE 小网格
# K=48,64 × r=16,32,48 × arch=2x512 × beta=0,1e-4,1e-3，epochs=30 → 18 组
# ============================================================

log ""
log "阶段 2/3: VAE 小网格 (18 组配置)"
log "K = [48, 64]"
log "r = [16, 32, 48]"
log "arch = [2x512]"
log "beta = [0.0, 0.0001, 0.001]"
log "epochs = 30"
log ""

START_TIME=$(date +%s)

python3 compress_benchmark.py \
  --data "${DATA}" \
  --out "${OUTDIR}/stage2_vae_grid" \
  --methods VAE \
  --K 48 64 \
  --latent 16 32 48 \
  --vae-arch 2x512 \
  --beta 0.0 0.0001 0.001 \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --wq 1.0 --wdq 0.25 --wtau 0.1 \
  --seed 42 \
  2>&1 | tee -a "${LOGFILE}"

STAGE2_TIME=$(($(date +%s) - START_TIME))
log "阶段 2 完成，耗时: $((STAGE2_TIME / 60)) 分 $((STAGE2_TIME % 60)) 秒"

# ============================================================
# 阶段 3: 精炼测试
# 从阶段 2 中选择 4-6 组最优配置，升级到 3x1024 + epochs=60
# ============================================================

log ""
log "阶段 3/3: 精炼测试 (高容量模型)"
log "基于阶段 2 结果，选择以下配置进行精炼："
log "K = [48, 64]"
log "r = [32, 48]  (选择中等压缩比)"
log "arch = [3x1024]  (升级架构)"
log "beta = [0.0, 0.001]  (保留最佳 beta 值)"
log "epochs = 60"
log ""

START_TIME=$(date +%s)

python3 compress_benchmark.py \
  --data "${DATA}" \
  --out "${OUTDIR}/stage3_vae_refine" \
  --methods VAE \
  --K 48 64 \
  --latent 32 48 \
  --vae-arch 3x1024 \
  --beta 0.0 0.001 \
  --epochs 60 \
  --batch-size 128 \
  --lr 1e-3 \
  --wq 1.0 --wdq 0.25 --wtau 0.1 \
  --seed 42 \
  2>&1 | tee -a "${LOGFILE}"

STAGE3_TIME=$(($(date +%s) - START_TIME))
log "阶段 3 完成，耗时: $((STAGE3_TIME / 60)) 分 $((STAGE3_TIME % 60)) 秒"

# ============================================================
# 汇总结果
# ============================================================

log ""
log "=========================================="
log "所有测试完成！"
log "=========================================="

TOTAL_TIME=$((STAGE1_TIME + STAGE2_TIME + STAGE3_TIME))
log ""
log "时间统计："
log "  阶段 1 (PCA):      $((STAGE1_TIME / 60)) 分 $((STAGE1_TIME % 60)) 秒"
log "  阶段 2 (VAE 小):   $((STAGE2_TIME / 60)) 分 $((STAGE2_TIME % 60)) 秒"
log "  阶段 3 (VAE 精):   $((STAGE3_TIME / 60)) 分 $((STAGE3_TIME % 60)) 秒"
log "  总计:              $((TOTAL_TIME / 60)) 分 $((TOTAL_TIME % 60)) 秒"
log ""

# 合并所有结果到一个文件
log "合并结果..."
python3 - <<EOF
import json
import os
from pathlib import Path

# 读取所有阶段的结果
stages = ['stage1_pca_sweep', 'stage2_vae_grid', 'stage3_vae_refine']
all_results = []

for stage in stages:
    summary_file = Path('${OUTDIR}') / stage / 'results_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            results = json.load(f)
            for r in results:
                r['stage'] = stage
            all_results.extend(results)

# 保存合并结果
output_file = Path('${OUTDIR}') / 'all_results_combined.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"合并完成：{len(all_results)} 组结果 → {output_file}")

# 生成排名报告
all_results_sorted = sorted(all_results, key=lambda x: x['metrics']['wrmse'])

print("\n" + "="*70)
print("Top 10 配置 (按 WRMSE 排序)")
print("="*70)
for i, r in enumerate(all_results_sorted[:10], 1):
    method = r['method']
    K = r['K']
    latent = r['latent']
    wrmse = r['metrics']['wrmse']
    stage = r['stage']
    
    if method == 'PCA':
        evr = r['explained_variance_ratio']
        print(f"{i:2d}. {method} K={K:3d} r={latent:3d} | WRMSE={wrmse:.6f} EVR={evr:.4f} [{stage}]")
    else:
        arch = r['vae_arch']
        beta = r['beta']
        print(f"{i:2d}. {method} K={K:3d} r={latent:3d} {arch:7s} β={beta:.4f} | WRMSE={wrmse:.6f} [{stage}]")

# 保存排名报告
report_file = Path('${OUTDIR}') / 'ranking_report.txt'
with open(report_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("压缩基准测试结果排名\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"总配置数: {len(all_results)}\n")
    f.write(f"PCA 配置: {len([r for r in all_results if r['method'] == 'PCA'])}\n")
    f.write(f"VAE 配置: {len([r for r in all_results if r['method'] == 'VAE'])}\n\n")
    
    f.write("Top 20 配置 (按 WRMSE 排序)\n")
    f.write("="*70 + "\n")
    for i, r in enumerate(all_results_sorted[:20], 1):
        method = r['method']
        K = r['K']
        latent = r['latent']
        wrmse = r['metrics']['wrmse']
        rmse_q = r['metrics']['rmse_q']
        rmse_dq = r['metrics']['rmse_dq']
        rmse_tau = r['metrics']['rmse_tau']
        stage = r['stage']
        
        compression_ratio = (2040 * 7853 * 21) / (2040 * latent)
        
        if method == 'PCA':
            evr = r['explained_variance_ratio']
            f.write(f"{i:2d}. {method} K={K:3d} r={latent:3d}\n")
            f.write(f"    WRMSE={wrmse:.6f} | EVR={evr:.4f} | 压缩率={compression_ratio:.1f}x\n")
        else:
            arch = r['vae_arch']
            beta = r['beta']
            f.write(f"{i:2d}. {method} K={K:3d} r={latent:3d} {arch} β={beta:.4f}\n")
            f.write(f"    WRMSE={wrmse:.6f} | 压缩率={compression_ratio:.1f}x\n")
        
        f.write(f"    RMSE: q={rmse_q:.4f}, dq={rmse_dq:.4f}, tau={rmse_tau:.4f}\n")
        f.write(f"    来源: {stage}\n\n")

print(f"\n排名报告已保存到: {report_file}")
EOF

log ""
log "结果文件："
log "  - ${OUTDIR}/all_results_combined.json (所有结果汇总)"
log "  - ${OUTDIR}/ranking_report.txt (排名报告)"
log "  - ${OUTDIR}/stage1_pca_sweep/results_summary.json"
log "  - ${OUTDIR}/stage2_vae_grid/results_summary.json"
log "  - ${OUTDIR}/stage3_vae_refine/results_summary.json"
log ""
log "日志文件: ${LOGFILE}"
log ""
log "测试完成！可以查看 ${OUTDIR}/ranking_report.txt 了解最佳配置" 