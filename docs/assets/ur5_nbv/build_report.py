#!/usr/bin/env python3
"""Build the offline report using the project's ARIS Markdown renderer.

Run from any directory: python3 docs/assets/ur5_nbv/build_report.py
Edit the Markdown and dashboard.html, then rebuild; never edit generated HTML.
"""

import base64
import hashlib
import importlib.util
import mimetypes
import re
import tempfile
from pathlib import Path


def main():
    assets = Path(__file__).resolve().parent
    root = assets.parents[2]
    source = root / "docs/ur5_nbv_kinematics_report.md"
    output = source.with_suffix(".html")
    renderer = root / ".agents/skills/render-html/scripts/render_html.py"
    if not renderer.is_file():
        manifest = root / ".aris/installed-skills-codex.txt"
        if manifest.is_file():
            for line in manifest.read_text().splitlines():
                key, _, value = line.partition("\t")
                if key == "repo_root":
                    renderer = Path(value) / "skills/render-html/scripts/render_html.py"
                    break
    if not renderer.is_file():
        raise SystemExit("ARIS render-html helper missing; restore the project skill installation.")
    spec = importlib.util.spec_from_file_location("aris_report_renderer", renderer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TEMPLATES_DIR = assets
    with tempfile.TemporaryDirectory(prefix="ur5-report-") as temporary:
        intermediate = Path(temporary) / "report.html"
        result = module.main([
            str(source), "--template", "dashboard", "--offline", "--no-toc",
            "--out", str(intermediate), "--title", "UR5 × 主动视觉｜从信息收益到可执行视角",
        ])
        if result:
            raise SystemExit(result)
        page = intermediate.read_text(encoding="utf-8")
    def embed(match):
        relative = match.group(1)
        image = (source.parent / relative).resolve()
        if image.parent != assets:
            raise ValueError(f"Unexpected image path: {relative}")
        mime = mimetypes.guess_type(image.name)[0]
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{encoded}"'
    page = re.sub(r'src="(assets/ur5_nbv/[^"<>]+)"', embed, page)
    template_hash = hashlib.sha256((assets / "dashboard.html").read_bytes()).hexdigest()
    page = page.replace("<!-- TEMPLATE_DIGEST -->", f'<meta name="report:template-sha256" content="{template_hash}">')
    if re.search(r"\{\{[A-Z_]+\}\}", page):
        raise ValueError("Unresolved template placeholder")
    output.write_text(page, encoding="utf-8")
    print(f"Offline report: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
