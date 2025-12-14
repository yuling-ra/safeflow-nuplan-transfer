# untitled1-project

把 `Untitled1.ipynb` 自动转换为适合在 VS Code 中开发/调试/打包的 Python 项目。

## 结构
```
untitled1-project/
  .vscode/
  notebooks/Untitled1.ipynb
  scripts/run.py
  src/untitled1/main.py
  tests/test_smoke.py
  pyproject.toml
  requirements.txt
  README.md
```

## 快速开始
```bash
python3 -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -U pip -r requirements.txt
python scripts/run.py
```

或安装为包：
```bash
pip install -e .
untitled1-run
```

> 说明：Jupyter 的 `%/!` 魔法命令已被注释，避免脚本执行报错。
生成时间：2025-09-29T19:55:35.722899
