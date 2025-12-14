{
  "spec_version": "v1.0",
  "experiment_name": "pouring_overfit_single_sample",
  "random_seed": 42,
  "fps_or_dt": {
    "fps": null,
    "dt_seconds": null,
    "notes": "如果需要动力学特征可填写；本实验对 dt 不敏感，仅为固定帧序。"
  },
  "source_sample": {
    "path": "research/Riemannian_Flow/MMLfD-Tutorial_Modified/datasets/pouring_data/1_water_200.pkl",
    "report": {
      "traj_shape": [480, 4, 4],
      "keys": ["traj", "bottle_idx", "mug_idx", "offset", "radius", "height", "label", "text"],
      "traj_semantics": "每帧 4x4 齐次变换矩阵, T_k = [[R_k, t_k], [0, 0, 0, 1]]"
    }
  },
  "io_conventions": {
    "axis_convention": "Right-handed; rows-as-basis; Open3D::transform uses column-major homography, but numpy stores row-major; ensure T is applied consistently.",
    "frame_zero_alignment": "可选：左乘 T_0^{-1} 将首帧对齐为单位阵，以降低全局方差。",
    "rotation_projection": "所有 R_k 先用 3x3 SVD 投影到 SO(3)：R = U diag(1,1,det(UV^T)) V^T",
    "tolerances": {
      "det_R_positive_min": 0.0,
      "svd_eps": 1e-8,
      "near_pi_clip_rad": 3.13
    }
  },
  "common_artifacts": {
    "out_dir": "artifacts/pouring_overfit_single",
    "filenames": {
      "raw_to_clean_npz": "01_clean_absolute_se3.npz",
      "abs_r9t3_npz": "02_abs_R9_t3_seq.npz",
      "delta_se3_npz": "03_delta_se3_seq.npz",
      "whitening_stats_abs": "04_whiten_abs.json",
      "whitening_stats_delta": "05_whiten_delta.json",
      "fm_ckpt_abs": "ckpt_abs_r9t3.pt",
      "fm_ckpt_delta": "ckpt_delta_se3.pt",
      "recon_abs_npz": "11_recon_abs_r9t3_seq.npz",
      "recon_delta_npz": "12_recon_delta_se3_seq.npz",
      "recon_abs_as_T_npz": "21_recon_abs_T_seq.npz",
      "recon_delta_as_T_npz": "22_recon_delta_T_seq.npz",
      "viz_abs_mp4": "31_viz_abs.mp4",
      "viz_delta_mp4": "32_viz_delta.mp4",
      "metrics_json": "41_metrics.json",
      "sanity_pngs_dir": "sanity_frames/"
    }
  },
  "method_R9_plus_t3": {
    "description": "绝对位姿序列，旋转用 9D 矩阵表示，训练/推理时通过批量 SVD 投影回 SO(3)。简单直接、工程最稳。",
    "preprocess": {
      "inputs": "T_seq ∈ R^{480×4×4} from pkl['traj']",
      "steps": [
        "对每帧 T_k 切出 R_k ∈ R^{3×3}, t_k ∈ R^3",
        "对 R_k 做 3×3 SVD 投影到 SO(3): R̂_k = U diag(1,1,det(UV^T)) V^T",
        "（可选）左乘起点对齐：T'_k = T_0^{-1} T_k，从而 R'_0=I, t'_0=0",
        "构造 r9_k = vec(R̂'_k)（按列或按行展开，固定约定）",
        "构造样本 X_abs[k] = concat(r9_k, t'_k) ∈ R^{12}",
        "得到序列 X_abs ∈ R^{480×12}",
        "对平移与旋转分通道统计均值/方差：μ_R9, σ_R9 (按每个矩阵元素); μ_t3, σ_t3 (按每轴)",
        "白化：Z_abs[:, :9] = (r9 - μ_R9) / σ_R9; Z_abs[:, 9:12] = (t - μ_t3)/σ_t3",
        "保存 Z_abs 以及 μ/σ 到磁盘"
      ],
      "outputs": {
        "npz": {
          "path": "artifacts/pouring_overfit_single/02_abs_R9_t3_seq.npz",
          "arrays": {
            "Z_abs": [480, 12],
            "mask": [480],
            "meta": {
              "vec_order": "row-major or column-major; must be fixed",
              "aligned_to_T0": true
            }
          }
        },
        "whitening_stats": {
          "path": "artifacts/pouring_overfit_single/04_whiten_abs.json",
          "μ_R9": [9],
          "σ_R9": [9],
          "μ_t3": [3],
          "σ_t3": [3]
        }
      }
    },
    "fm_learning_target": {
      "what_model_learns": "在欧式空间 R^{480×12} 上的向量场/score/噪声回归（依据所选 FM 变体），旋转与平移可作为两个通道分别建模，并允许 cross-attention。",
      "input_tensor_shape": "B=1（过拟合单样本）, L=480, C=12（前 9 为旋转矩阵展开，后 3 为平移）",
      "conditioning": "可空；或添加起点/终点锚点、文本/标签等；本测试可设为空。"
    },
    "inference_and_reconstruction": {
      "sampling": "从先验噪声 Z0 ∈ R^{480×12} 通过 FM ODE/SDE 积分 → Ẑ_abs ∈ R^{480×12}",
      "dewhiten": "反白化：r9̂ = Ẑ_abs[:,:9]*σ_R9 + μ_R9; t̂ = Ẑ_abs[:,9:]*σ_t3 + μ_t3",
      "so3_projection": "把每帧的 r9̂ 重塑为 3×3 矩阵 R̃，SVD 投影到 SO(3)：R̂ = U diag(1,1,det(UV^T)) V^T",
      "rebuild_T": "T̂'_k = [[R̂_k, t̂_k],[0,0,0,1]]，若之前做了 T0 对齐，则 T̂_k = T_0 · T̂'_k",
      "outputs": {
        "npz_seq": {
          "path": "artifacts/pouring_overfit_single/21_recon_abs_T_seq.npz",
          "arrays": {
            "T_hat": [480, 4, 4]
          }
        }
      }
    },
    "sanity_checks": [
      "∀k: |det(R̂_k) - 1| < 1e-4",
      "最后一行始终 [0,0,0,1]",
      "若对齐到 T0，检查起点误差 ||log(T̂_0)|| ≈ 0"
    ],
    "pros_cons": {
      "pros": ["实现最短", "每帧独立投影，几乎不爆数", "无需连乘"],
      "cons": ["旋转 9D 冗余，优化表面略复杂", "不显式利用小步变化的局部线性化"]
    }
  },
  "method_delta_se3": {
    "description": "相邻帧增量在李代数 se(3)（6D）中建模，学习小量更线性、梯度更均匀。推理后用群指数映射与连乘重建绝对位姿。",
    "preprocess": {
      "inputs": "T_seq ∈ R^{480×4×4} from pkl['traj']",
      "steps": [
        "同 R9+t3，对所有 R_k 做 SVD 投影到 SO(3)",
        "（可选）左乘起点对齐：T'_k = T_0^{-1} T_k",
        "构造相邻增量：ΔT_k = (T'_{k-1})^{-1} T'_k,  k=1..479",
        "对每个 ΔT_k 计算李代数坐标 ξ_k = log(ΔT_k) ∈ R^6, 记为 ξ_k = [ω_k(3), v_k(3)]",
        "分别对白化旋转与平移增量：μ_ω,σ_ω ∈ R^3; μ_v,σ_v ∈ R^3",
        "得到 Z_delta[k] = concat( (ω_k-μ_ω)/σ_ω , (v_k-μ_v)/σ_v ) ∈ R^6",
        "保存 Z_delta ∈ R^{479×6} 与白化统计"
      ],
      "outputs": {
        "npz": {
          "path": "artifacts/pouring_overfit_single/03_delta_se3_seq.npz",
          "arrays": {
            "Z_delta": [479, 6],
            "mask": [479],
            "meta": {
              "aligned_to_T0": true,
              "se3_basis": "twist coordinate, left-trivialized",
              "log_exp_impl": "Rodrigues for so(3), closed-form V for se(3)"
            }
          }
        },
        "whitening_stats": {
          "path": "artifacts/pouring_overfit_single/05_whiten_delta.json",
          "μ_ω": [3],
          "σ_ω": [3],
          "μ_v": [3],
          "σ_v": [3]
        }
      }
    },
    "fm_learning_target": {
      "what_model_learns": "在欧式空间 R^{479×6} 上的向量场/score/噪声回归（按 FM 变体），建议把 ω 与 v 分两个通道建模并允许 cross-attention。",
      "input_tensor_shape": "B=1, L=479, C=6（3 旋转增量 + 3 平移增量）",
      "conditioning": "可选：提供首帧 T'_0=I（对齐已完成），或绝对目标帧；本过拟合测试可空。"
    },
    "inference_and_reconstruction": {
      "sampling": "从先验噪声 Z0 ∈ R^{479×6} 经 FM ODE/SDE → Ẑ_delta ∈ R^{479×6}",
      "dewhiten": "ω̂ = Ẑ_delta[:,:3]*σ_ω + μ_ω; v̂ = Ẑ_delta[:,3:]*σ_v + μ_v",
      "exp_map": "对每步 ξ̂_k=[ω̂_k, v̂_k] 调用 se(3) 指数映射得到 ΔT̂_k = exp(ξ̂_k^)",
      "chain_rebuild": "T̂'_0 = I_4;  对 k=1..479:  T̂'_k = T̂'_{k-1} · ΔT̂_k",
      "optional_projection": "每步重建后对 R̂'_k 做一次 SVD 投影，防止累计漂移",
      "restore_absolute": "若预处理对齐过：T̂_k = T_0 · T̂'_k；否则 T̂_k = T̂'_k",
      "outputs": {
        "npz_seq": {
          "path": "artifacts/pouring_overfit_single/22_recon_delta_T_seq.npz",
          "arrays": {
            "T_hat": [480, 4, 4]
          }
        }
      }
    },
    "sanity_checks": [
      "∀k: ||ω̂_k|| < near_pi_clip_rad（通常远小于 π）",
      "连乘后的 R̂_k 进行 det≈1 检查，必要时 SVD 纠正",
      "端点误差 ||log((T̂_479)^{-1} T_479)|| 合理变小（过拟合应趋近 0）"
    ],
    "pros_cons": {
      "pros": ["小步增量，近似线性，梯度均匀", "天然匹配流形向量场的几何语义"],
      "cons": ["需要连乘重建绝对位姿", "实现上多一步 log/exp"]
    }
  },
  "hybrid_recommendation": {
    "status": "optional",
    "idea": "用绝对 R9+t3 作为上下文/条件输入，用 Δse(3) 作为 FM 的主要学习变量与采样对象。",
    "benefit": "兼得全局稳健（R9+t3）与局部线性（Δse(3)），对长序列更稳。",
    "cost": "实现稍复杂，但推理/训练吞吐接近 Δse(3)。"
  },
  "evaluation_and_visualization": {
    "metrics": {
      "per_frame": [
        "rot_geodesic_deg(k) = acos((trace(R_k^T R̂_k)-1)/2) * 180/pi",
        "trans_l2(k) = ||t_k - t̂_k||_2"
      ],
      "trajectory_level": [
        "mean_rot_deg, max_rot_deg, p95_rot_deg",
        "mean_trans_cm, max_trans_cm, p95_trans_cm",
        "endpoint_SE3_error = ||log(T̂_479^{-1} T_479)|| (旋转/平移分量可加权)",
        "smoothness: finite-difference jerk on t̂/ω̂（可选）"
      ]
    },
    "success_criteria": "过拟合测试下，均值/端点误差应接近 0；可允许极小数值噪声。",
    "visualization": {
      "open3d_apply": "mesh.transform(T̂_k)",
      "exports": ["mp4", "frame_pngs"],
      "overlays": ["显示坐标轴、误差曲线随帧绘制"]
    }
  },
  "failure_modes_and_handlers": {
    "near_colinearity_R9": "对 R9 不适用（该法不需要 Gram-Schmidt），SVD 投影能稳定修正。",
    "delta_pi_neighborhood": "若个别 ΔR 接近 π，建议预平滑或分段；当前倒水数据通常不会触发。",
    "drift_in_chain": "Δse(3) 路线可在每步后做 SVD 投影；必要时每 N 帧与锚点重定位。",
    "whitening_mismatch": "确保推理使用与训练同一 μ/σ；存为 JSON 固化。"
  }
}
