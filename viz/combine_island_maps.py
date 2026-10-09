"""
Composite the three per-island map figures (viz/plot_island_maps.py's
output, run with --no-inset) into one 7x10in figure for the paper --
replaces three separate Figure 12/13/14 floats with one, per review
feedback asking for exactly that.

LAYOUT: a vertical stack, one island after another, each given a cell of
the SAME fixed aspect ratio -- but each island's own map is placed into
its cell at its own natural scale, not stretched or forced to that
aspect ratio itself. An earlier version of this script tried forcing the
underlying zoom windows in plot_island_maps.py to a single aspect ratio
so the saved PNGs would already be identically shaped; that distorted
the maps themselves (padding narrow islands out with a lot of mostly-
empty extra width to hit the target ratio, making the geography look
unnaturally stretched). The real cut geometry is kept exactly as
plot_island_maps.py naturally renders it -- the "same aspect ratio" is
purely a property of the three cells on the page, not of the map content
inside them. Each image is fit into its cell preserving its own true
aspect ratio (shrink to fit, centered -- "letterboxed" with white on
whichever axis doesn't fill, never cropped or distorted -- these are
geographic maps, so distorting aspect ratio would misrepresent real
distances).

Run viz/plot_island_maps.py --island a g o --no-inset first (the locator
inset is dropped deliberately -- see that script's --no-inset help text).

Usage:
    python viz/plot_island_maps.py --island a g o --no-inset
    python viz/combine_island_maps.py
"""

import argparse

from PIL import Image, ImageDraw, ImageFont

from configs import setup_logger
from configs.virginia_config import VA_FIGURES_DIR
from viz.plot_island_maps import PAPER_ISLAND_LABELS
from viz.plot_utils import pil_save_with_retry

logger = setup_logger(log_file="logs/combine_island_maps.log")

DPI = 300
FIG_W_IN = 7.0
FIG_H_IN = 10.0
GAP_PX = 30
LABEL_H_PX = 50
LABEL_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL_FONT_SIZE = 32

# Shared cell aspect ratio (width / height) every island's panel is fit
# into, regardless of that island's own true aspect ratio. 1.0 (a square
# cell) is a reasonable middle ground for this project's islands, which
# range from notably portrait to notably landscape -- a square wastes the
# least space on average across all of them. Change this if a future
# island set skews consistently one way.
CELL_ASPECT = 1.0


def _load_label_font():
    try:
        return ImageFont.truetype(LABEL_FONT_PATH, LABEL_FONT_SIZE)
    except Exception:
        logger.warning(f"Could not load {LABEL_FONT_PATH}; falling back to PIL's default font.")
        return ImageFont.load_default()


def _fit_and_paste(canvas, img, cell_x0, cell_y0, cell_w, cell_h):
    """Shrink img to fit within (cell_w, cell_h) preserving its own
    aspect ratio, and paste it centered in that cell -- letterboxed with
    white, never cropped or distorted."""
    scale = min(cell_w / img.width, cell_h / img.height)
    w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    resized = img.resize((w, h), Image.LANCZOS)
    x = cell_x0 + (cell_w - w) // 2
    y = cell_y0 + (cell_h - h) // 2
    canvas.paste(resized, (x, y))


def build_layout(paths, labels, out_path):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    n = len(imgs)

    canvas_w = int(FIG_W_IN * DPI)
    max_canvas_h = int(FIG_H_IN * DPI)

    # Every cell is the same shape (CELL_ASPECT), so every cell's natural
    # height at full canvas width is the same too.
    natural_cell_h = canvas_w / CELL_ASPECT
    overhead = n * LABEL_H_PX + (n - 1) * GAP_PX
    natural_image_h = n * natural_cell_h

    scale = min(1.0, (max_canvas_h - overhead) / natural_image_h)
    cell_h = int(natural_cell_h * scale)
    canvas_h = n * cell_h + overhead

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_label_font()

    y = 0
    for img, label in zip(imgs, labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((canvas_w - tw) / 2, y + 6), label, fill="black", font=font)
        y += LABEL_H_PX
        _fit_and_paste(canvas, img, 0, y, canvas_w, cell_h)
        y += cell_h + GAP_PX

    pil_save_with_retry(canvas, out_path, dpi=(DPI, DPI))
    logger.info(
        f"Saved {out_path} ({canvas_w/DPI:.2f}in x {canvas_h/DPI:.2f}in"
        f"{', scaled to fit 7x10in budget' if scale < 1.0 else ''})"
    )


def main(island_ids):
    paths = [VA_FIGURES_DIR / f"va_island_{PAPER_ISLAND_LABELS.get(i, i.upper()).lower()}_map.png" for i in island_ids]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run `python viz/plot_island_maps.py --island "
            f"{' '.join(island_ids)} --no-inset` first."
        )
    labels = [f"({chr(ord('a') + i)}) Island {PAPER_ISLAND_LABELS.get(iid, iid.upper())}" for i, iid in enumerate(island_ids)]
    out_path = VA_FIGURES_DIR / "va_island_maps_combined.png"
    build_layout(paths, labels, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine the per-island map figures into one 7x10in figure, stacked vertically in same-shaped cells.")
    parser.add_argument("--island", type=str, nargs="+", default=["a", "g", "o"])
    args = parser.parse_args()
    main(island_ids=args.island)
