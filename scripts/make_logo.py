#!/usr/bin/env -S uv run --script
"""Draw the ksay logo and write every size of it to docs/logo.

    uv run scripts/make_logo.py

The mark is a terminal prompt followed by a waveform: type a command, hear a voice.
The wordmark is set in JetBrains Mono ExtraBold (SIL Open Font License 1.1), turned
into outlines, so the logo needs no font to display. The font is downloaded once into
~/.cache/kokoro-say, with its checksum checked.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["fonttools", "resvg-py"]
# ///

from __future__ import annotations

import hashlib
import math
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "logo"
CACHE = Path.home() / ".cache" / "kokoro-say" / "fonts"
FONT_ZIP = (
    "https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/"
    "JetBrainsMono-2.304.zip"
)
FONT_SHA256 = "6f6376c6ed2960ea8a963cd7387ec9d76e3f629125bc33d1fdcd7eb7012f7bbf"
EXTRA_BOLD = "fonts/ttf/JetBrainsMono-ExtraBold.ttf"
MEDIUM = "fonts/ttf/JetBrainsMono-Medium.ttf"

GREEN = "#3fb950"
BLUE_TOP, BLUE_BOTTOM = "#79c0ff", "#388bfd"
INK_ON_DARK, INK_ON_LIGHT = "#f0f6fc", "#1f2328"
BARS = [50, 108, 160, 90, 44]  # heights of the waveform bars in the 256-unit mark
CHEVRON_W, CHEVRON_H, CHEVRON_STROKE = 74, 92, 24
BAR_W, BAR_PITCH, GAP = 17, 25, 14


def font(name: str) -> Path:
    """A font file from the JetBrains Mono release, downloaded on first use."""
    path = CACHE / Path(name).name
    if not path.exists():
        archive = CACHE / "JetBrainsMono.zip"
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FONT_ZIP) as response:
            archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != FONT_SHA256:
            archive.unlink()
            raise SystemExit("the downloaded font archive failed its checksum")
        with zipfile.ZipFile(archive) as bundle:
            for member in (EXTRA_BOLD, MEDIUM):
                (CACHE / Path(member).name).write_bytes(bundle.read(member))
    return path


def letters(
    text: str, face: Path, size: float, x: float, baseline: float, tracking: int
):
    """One SVG path per letter, and the width of the whole text."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    ttf = TTFont(face)
    glyphs, cmap = ttf.getGlyphSet(), ttf.getBestCmap()
    scale = size / ttf["head"].unitsPerEm
    paths, cursor = [], 0
    for char in text:
        glyph = glyphs[cmap[ord(char)]]
        pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
        glyph.draw(
            TransformPen(pen, (scale, 0, 0, -scale, x + cursor * scale, baseline))
        )
        paths.append(pen.getCommands())
        cursor += glyph.width + tracking
    return paths, (cursor - tracking) * scale


def mark(uid: str) -> str:
    """The icon, drawn in a 256 by 256 box."""
    bars_w = BAR_PITCH * (len(BARS) - 1) + BAR_W
    left = 128 - (CHEVRON_W + GAP + bars_w) / 2
    x0 = left + CHEVRON_STROKE / 2
    top, bottom = 128 - CHEVRON_H / 2, 128 + CHEVRON_H / 2
    tip = x0 + CHEVRON_W - CHEVRON_STROKE
    chevron = f"M{x0:.1f} {top:.1f}L{tip:.1f} 128L{x0:.1f} {bottom:.1f}"
    bar_x = left + CHEVRON_W + GAP
    bars = "".join(
        f'<rect x="{bar_x + i * BAR_PITCH:.1f}" y="{128 - h / 2:.1f}" width="{BAR_W}" '
        f'height="{h}" rx="{BAR_W / 2}"/>'
        for i, h in enumerate(BARS)
    )
    return (
        "<defs>"
        f'<linearGradient id="{uid}-bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#263247"/><stop offset="1" stop-color="#0b0f14"/>'
        "</linearGradient>"
        f'<linearGradient id="{uid}-wave" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{BLUE_TOP}"/>'
        f'<stop offset="1" stop-color="{BLUE_BOTTOM}"/>'
        "</linearGradient></defs>"
        f'<rect x="6" y="6" width="244" height="244" rx="58" fill="url(#{uid}-bg)"/>'
        '<rect x="7" y="7" width="242" height="242" rx="57" fill="none" '
        'stroke="#fff" stroke-opacity=".2" stroke-width="2"/>'
        f'<path d="{chevron}" fill="none" stroke="{GREEN}" '
        f'stroke-width="{CHEVRON_STROKE}" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        f'<g fill="url(#{uid}-wave)">{bars}</g>'
    )


def lockup(ink: str, uid: str) -> tuple[str, float, float]:
    """The icon with the wordmark beside it: a group, its width and its height."""
    icon, size, gap = 176, 168, 20
    baseline = icon / 2 + 0.55 * size / 2  # the x-height is centred on the icon
    paths, width = letters("ksay", font(EXTRA_BOLD), size, icon + gap, baseline, -8)
    k, *say_letters = paths
    group = (
        f'<g transform="scale({icon / 256})">{mark(uid)}</g>'
        f'<path d="{k}" fill="{GREEN}"/><path d="{"".join(say_letters)}" fill="{ink}"/>'
    )
    return group, icon + gap + width, icon


def svg(body: str, width: float, height: float, pad: float = 0) -> str:
    w, h = width + 2 * pad, height + 2 * pad
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.1f} {h:.1f}">'
        f'<g transform="translate({pad} {pad})">{body}</g>'
        "</svg>\n"
    )


def social_preview() -> str:
    """The 1280 by 640 image GitHub shows when the repository is shared."""
    group, width, _ = lockup(INK_ON_DARK, "social")
    scale = 1.45
    body = ['<rect width="1280" height="640" fill="#0d1117"/>']
    for i in range(64):  # a faint waveform along the bottom
        h = 18 + 70 * abs(math.sin(i * 0.37) * math.sin(i * 0.11 + 1))
        body.append(
            f'<rect x="{i * 20 + 5}" y="{620 - h:.1f}" width="10" height="{h:.1f}" '
            f'rx="5" fill="{BLUE_BOTTOM}" fill-opacity=".16"/>'
        )
    body.append(
        f'<g transform="translate({(1280 - width * scale) / 2:.1f} 96) '
        f'scale({scale})">{group}</g>'
    )
    face = font(MEDIUM)
    for text, size, color, baseline in (
        ("Natural-sounding text-to-speech for your terminal", 34, "#c9d1d9", 440),
        ("Offline · Kokoro-82M · macOS · Linux · Windows", 26, "#8b949e", 500),
    ):
        text_width = letters(text, face, size, 0, 0, 0)[1]
        paths, _ = letters(text, face, size, (1280 - text_width) / 2, baseline, 0)
        body.append(f'<path d="{"".join(paths)}" fill="{color}"/>')
    return svg("".join(body), 1280, 640)


def main() -> int:
    import resvg_py

    def png(text: str, width: int) -> bytes:
        return bytes(resvg_py.svg_to_bytes(svg_string=text, width=width))

    OUT.mkdir(parents=True, exist_ok=True)
    icon = svg(mark("mark"), 256, 256)
    files: dict[str, str | bytes] = {"ksay-mark.svg": icon}
    for theme, ink in (("dark", INK_ON_DARK), ("light", INK_ON_LIGHT)):
        group, width, height = lockup(ink, theme)
        text = svg(group, width, height, pad=8)
        files[f"ksay-logo-{theme}.svg"] = text
        files[f"ksay-logo-{theme}.png"] = png(text, 1200)
    files["ksay-mark-512.png"] = png(icon, 512)
    files["social-preview.png"] = png(social_preview(), 1280)
    for name, content in files.items():
        data = content if isinstance(content, bytes) else content.encode()
        (OUT / name).write_bytes(data)
        print(f"docs/logo/{name}  {len(data) / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
