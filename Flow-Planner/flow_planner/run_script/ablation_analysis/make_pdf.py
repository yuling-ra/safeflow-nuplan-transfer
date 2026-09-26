#!/usr/bin/env python3
"""Render the Markdown ablation report as a self-contained printable HTML."""

from pathlib import Path

import markdown


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ablation_analysis_report.md"
HTML = HERE / "ablation_analysis_report.html"

body = markdown.markdown(
    SOURCE.read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code"],
)

document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>SafeFlow、Flow-Planner 与 PDM 消融实验分析</title>
<style>
@page {{ size: A4; margin: 16mm 15mm 17mm; }}
* {{ box-sizing: border-box; }}
body {{
  font-family: "Noto Sans CJK SC", "Noto Sans CJK TC", "Microsoft YaHei", sans-serif;
  color: #20252b; font-size: 10.5pt; line-height: 1.55; margin: 0;
}}
h1 {{ color: #163f4a; font-size: 23pt; line-height: 1.2; margin: 0 0 8pt; }}
h2 {{ color: #166b72; font-size: 16pt; border-bottom: 1.3px solid #c8d5d8; padding-bottom: 3pt; margin: 18pt 0 8pt; }}
h3 {{ color: #315c63; font-size: 12.5pt; margin: 13pt 0 5pt; }}
p {{ margin: 5pt 0 7pt; }}
li {{ margin: 2pt 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 7pt 0 11pt; page-break-inside: avoid; }}
th, td {{ border: 0.5px solid #aebdc1; padding: 4pt 5pt; text-align: center; }}
th {{ background: #e7f0f0; color: #153e46; font-weight: 700; }}
td:first-child, th:first-child {{ text-align: left; }}
img {{ display: block; max-width: 100%; max-height: 125mm; margin: 8pt auto 3pt; page-break-inside: avoid; }}
blockquote {{ border-left: 3px solid #167c80; background: #f2f7f7; margin: 8pt 0; padding: 6pt 10pt; }}
code {{ background: #f0f2f3; padding: 1pt 3pt; font-family: "Noto Sans Mono", monospace; }}
pre {{ background: #f0f2f3; padding: 7pt; white-space: pre-wrap; font-size: 8.5pt; }}
hr {{ border: 0; border-top: 1px solid #d9e1e2; }}
a {{ color: #126e78; }}
h2, h3, table, blockquote {{ break-inside: avoid; }}
</style>
</head>
<body>{body}</body>
</html>"""

HTML.write_text(document, encoding="utf-8")
print(HTML)
