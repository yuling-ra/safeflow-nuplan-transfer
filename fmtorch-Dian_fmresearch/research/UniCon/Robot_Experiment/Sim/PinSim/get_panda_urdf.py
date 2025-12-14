#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from pathlib import Path
from robot_descriptions import panda_description as panda
import argparse

PKG_URI_RE = re.compile(r'package://([^/]+)/(.+)$')

def main():
    parser = argparse.ArgumentParser(description="Export panda_description (URDF + meshes) into a local project folder.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory (default: current dir)")
    parser.add_argument("--name", type=str, default="panda_description", help="Subfolder name to create (default: panda_description)")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    target_root = outdir / args.name
    target_urdf_dir = target_root / "urdf"
    target_mesh_dir = target_root / "meshes"

    src_repo = Path(panda.REPOSITORY_PATH).resolve()
    src_urdf = Path(panda.URDF_PATH).resolve()

    # 典型结构： .../example-robot-data/robots/panda_description/{urdf, meshes}
    # 找到 panda_description 根
    # 优先用 URDF 路径倒推
    src_pd_root = src_urdf.parent.parent  # .../panda_description
    if src_pd_root.name != "panda_description":
        # 保险：从常见路径推断
        candidate = src_repo / "robots" / "panda_description"
        if candidate.is_dir():
            src_pd_root = candidate

    src_mesh_dir = src_pd_root / "meshes"
    src_visual = src_mesh_dir / "visual"
    src_collision = src_mesh_dir / "collision"

    print("=== Source ===")
    print("repo: ", src_repo)
    print("urdf: ", src_urdf)
    print("pd_root:", src_pd_root)
    print("meshes:", src_mesh_dir)

    # 创建目标目录
    target_urdf_dir.mkdir(parents=True, exist_ok=True)
    (target_mesh_dir / "visual").mkdir(parents=True, exist_ok=True)
    (target_mesh_dir / "collision").mkdir(parents=True, exist_ok=True)

    # 复制 URDF
    target_urdf = target_urdf_dir / src_urdf.name
    shutil.copy2(src_urdf, target_urdf)

    # 读取并解析 URDF 中的 mesh 引用
    text = target_urdf.read_text(encoding="utf-8")

    def pkg_to_local(match):
        """
        将 package://<pkg>/<path> 转换成本地的相对路径 "meshes/..."
        逻辑：如果原路径包含 .../panda_description/meshes/xxx
             就裁剪到 "meshes/xxx"
        """
        full = match.group(2)  # <path> 部分
        # 规范化为 Path 再字符串化，便于查找子串
        p = Path(full)
        try:
            # 查找 "panda_description/meshes/" 的位置
            idx = full.split("/").index("panda_description")
            # 重组为从 meshes 起的相对段
            rel_after_pd = "/".join(full.split("/")[idx+1:])  # e.g. "meshes/visual/link0.dae"
            if rel_after_pd.startswith("meshes/"):
                return 'filename="' + rel_after_pd + '"'
        except ValueError:
            pass
        # 兜底：如果路径里直接包含 "meshes/"
        if "meshes/" in full:
            rel = "meshes/" + full.split("meshes/")[1]
            return 'filename="' + rel + '"'
        # 再兜底：不变化（保持原样）
        return 'filename="' + match.group(2) + '"'

    # 把所有 filename="package://.../..." 替换为 filename="meshes/..."
    new_text = re.sub(r'filename="package://([^/]+)/([^"]+)"', pkg_to_local, text)

    # 把小概率存在的绝对路径/其他写法也尝试替换为相对 meshes（可选）
    # new_text = new_text.replace("package://example-robot-data/robots/panda_description/", "")

    target_urdf.write_text(new_text, encoding="utf-8")

    # 收集 URDF 中引用到的 mesh 文件并复制
    mesh_paths = re.findall(r'filename="([^"]+)"', new_text)
    copied = set()
    missing = []

    for m in mesh_paths:
        if not m.startswith("meshes/"):
            # 忽略非 mesh 引用（例如某些特殊相对路径）
            continue
        rel = Path(m)  # meshes/visual/xxx / meshes/collision/xxx
        src = src_pd_root / rel
        dst = target_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            if dst.as_posix() not in copied:
                shutil.copy2(src, dst)
                copied.add(dst.as_posix())
        else:
            missing.append(src)

    print("\n=== Export Result ===")
    print("Exported to:", target_root)
    print("URDF:", target_urdf)
    print("Meshes copied:", len(copied))
    if missing:
        print("Missing meshes (not found at source):")
        for p in missing:
            print(" -", p)

    # 额外：把一个“已解析为相对路径”的副本再写一份（可选）
    resolved_urdf = target_urdf_dir / (target_urdf.stem + "_local.urdf")
    resolved_urdf.write_text(new_text, encoding="utf-8")
    print("Local-path URDF:", resolved_urdf)

    print("\nDone ✅ 现在你可以在项目中用如下相对路径加载：")
    print(f"  {args.name}/urdf/{resolved_urdf.name}")

if __name__ == "__main__":
    main()
