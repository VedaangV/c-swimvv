"""
Composite the three per-island map figures (viz/plot_island_maps.py's
output, run with --no-inset) into one 7x10in figure for the paper --
replaces three separate Figure 12/13/14 floats with one, per review
feedback asking for exactly that.

LAYOUT: a simple vertical stack, one island after another, each at the
same width. This only works because plot_island_maps.py now forces every
island's zoom window to the same ZOOM_TARGET_ASPECT -- earlier versions
of this script had to deal with each island's real cut geometry
producing a very different natural aspect ratio per panel (one tall, one
wide), which needed increasingly elaborate per-panel sizing logic and
still left some panels far smaller than others. With all three source
images now sharing one fixed aspect ratio, they stack cleanly at a
common width with no panel-specific layout math at all.

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

logger = setup_logger(log_file="logs/combine_island_maps.log")

DPI = 300
FIG_W_IN = 7.0
FIG_H_IN = 10.0
GAP_PX = 30
LABEL_H_PX = 50
LABEL_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL_FONT_SIZE = 32


def _load_label_font():
    try:
        return ImageFont.truetype(LABEL_FONT_PATH, LABEL_FONT_SIZE)
    except Exception:
        logger.warning(f"Could not load {LABEL_FONT_PATH}; falling back to PIL's default font.")
        return ImageFont.load_default()


def build_layout(paths, labels, out_path):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    n = len(imgs)

    canvas_w = int(FIG_W_IN * DPI)
    max_canvas_h = int(FIG_H_IN * DPI)

    # Every image is scaled to the same width; since plot_island_maps.py
    # now forces them all to the same aspect ratio, this also makes them
    # all the same height -- no letterboxing or per-panel sizing needed.
    row_h = [int(canvas_w / (img.width / img.height)) for img in imgs]
    overhead = n * LABEL_H_PX + (n - 1) * GAP_PX
    natural_image_h = sum(row_h)

    scale = min(1.0, (max_canvas_h - overhead) / natural_image_h)
    row_h = [int(h * scale) for h in row_h]
    canvas_h = sum(row_h) + overhead

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_label_font()

    y = 0
    for img, label, h in zip(imgs, labels, row_h):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((canvas_w - tw) / 2, y + 6), label, fill="black", font=font)
        y += LABEL_H_PX
        resized = img.resize((canvas_w, h), Image.LANCZOS)
        canvas.paste(resized, (0, y))
        y += h + GAP_PX

    canvas.save(out_path, dpi=(DPI, DPI))
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
    parser = argparse.ArgumentParser(description="Combine the per-island map figures into one 7x10in figure, stacked vertically.")
    parser.add_argument("--island", type=str, nargs="+", default=["a", "g", "o"])
    args = parser.parse_args()
    main(island_ids=args.island)
