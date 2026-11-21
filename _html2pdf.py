"""Convert nbconvert HTML export to PDF via Playwright Chromium."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()

PRINT_CSS = """
<style>
@page { size: A4; margin: 13mm 11mm 16mm 11mm; }
html, body { font-family: "Microsoft YaHei", "Segoe UI", "Helvetica Neue", sans-serif; }
body { font-size: 10.5pt; line-height: 1.55; }
/* 去掉 notebook 的页面内边距 */
.jp-Notebook, .notebook { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
.jp-Cell, .cell { padding: 4px 0 !important; }
/* 代码块：自动换行，避免长行被裁掉 */
pre, .jp-InputArea-editor, .highlight pre, .jp-OutputArea pre {
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
  overflow: visible !important;
  font-size: 8.2pt !important;
  line-height: 1.35 !important;
  margin: 4px 0 !important;
}
.jp-InputArea-editor, .jp-OutputArea pre { border-left: 3px solid #dfe4ea !important; padding: 6px 9px !important; }
/* 图片 / 输出 / 表格：尽量不跨页断开 */
.jp-OutputArea, .output_area, .output_wrapper, .jp-RenderedImage,
img, table, .jp-RenderedHTMLCommon table { page-break-inside: avoid; break-inside: avoid; }
img { max-width: 100% !important; height: auto !important; }
/* 标题：不与正文分离 */
h1, h2, h3, h4 { page-break-after: avoid; break-after: avoid; }
h1 { font-size: 19pt; } h2 { font-size: 15pt; } h3 { font-size: 12.5pt; }
/* 收起代码块的 prompt 数字，减少噪声 */
.jp-InputPrompt, .prompt { color: #9aa4b2 !important; font-size: 8pt !important; }
/* 表格排版 */
table { font-size: 8.5pt !important; max-width: 100% !important; }
th, td { padding: 3px 6px !important; }
/* 打印时隐藏浮动/交互元素 */
.toolbar, #maintoolbar { display: none !important; }
.jp-RenderedHTMLCommon { max-width: 100% !important; overflow: hidden !important; }
</style>
"""

html = src.read_text(encoding="utf-8")
if "</head>" in html:
    html = html.replace("</head>", PRINT_CSS + "</head>", 1)
else:
    html = PRINT_CSS + html
src.write_text(html, encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(src.as_uri(), wait_until="load", timeout=180000)
    page.wait_for_timeout(2500)  # let MathJax / fonts settle
    page.emulate_media(media="print")
    page.pdf(
        path=str(dst),
        format="A4",
        print_background=True,
        margin={"top": "13mm", "bottom": "16mm", "left": "11mm", "right": "11mm"},
        display_header_footer=True,
        header_template="<div></div>",
        footer_template=(
            '<div style="width:100%;font-size:7.5pt;color:#8a94a6;'
            'font-family:Segoe UI,sans-serif;padding:0 11mm;display:flex;'
            'justify-content:space-between;">'
            '<span>rock_fall_001</span>'
            '<span class="pageNumber"></span>/<span class="totalPages"></span>'
            '</div>'
        ),
        prefer_css_page_size=True,
    )
    browser.close()

print("PDF:", dst, dst.stat().st_size, "bytes")
