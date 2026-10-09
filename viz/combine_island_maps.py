"""
Composite the three per-island map figures (viz/plot_island_maps.py's
output, run with --no-inset) into one vertical-stack figure for the paper
-- replaces three separate Figure 12/13/14 floats with one, per review
feedback asking for exactly that.

Vertical stack, not side-by-side: the three island maps have very
different native aspect ratios (Island B's map in particular is much
wider/flatter than A or C), so laying them out side-by-side at a common
height crushes all three down to a sliver. Stacking at a common width
instead lets each panel keep a reasonable height.

Run viz/plot_island_maps.py --island a g o --no-inset first (the
locator inset is dropped here deliberately -- all three islands cluster
in the same central-Virginia/Richmond area, so repeating a near-identical
"Location in Virginia" panel three times in one combined figure added
nothing a reader didn't already get from the first repetition).

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
MAX_W_IN = 7.0
MAX_H_IN = 10.0
GAP_PX = 30
LABEL_H_PX = 55
LABEL_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL_FONT_SIZE = 34


def _load_label_font():
    try:
        return ImageFont.truetype(LABEL_FONT_PATH, LABEL_FONT_SIZE)
    except Exception:
        logger.warning(f"Could not load {LABEL_FONT_PATH}; falling back to PIL's default font.")
        return ImageFont.load_default()


def build_vstack(paths, labels, out_path, max_w_in: float = MAX_W_IN, max_h_in: float = MAX_H_IN):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    n = len(imgs)
    aspects = [im.width / im.height for im in imgs]
    max_h_px = max_h_in * DPI
    overhead = GAP_PX * (n - 1) + LABEL_H_PX * n
    width_px = (max_h_px - overhead) / sum(1.0 / a for a in aspects)
    width_px = min(width_px, max_w_in * DPI)
    heights = [int(width_px / a) for a in aspects]

    font = _load_label_font()
    canvas_w = int(width_px)
    canvas_h = sum(heights) + GAP_PX * (n - 1) + LABEL_H_PX * n
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    y = 0
    for im, h, label in zip(imgs, heights, labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((canvas_w - tw) / 2, y + 8), label, fill="black", font=font)
        y += LABEL_H_PX
        canvas.paste(im.resize((canvas_w, h), Image.LANCZOS), (0, y))
        y += h + GAP_PX

    canvas.save(out_path, dpi=(DPI, DPI))
    logger.info(f"Saved {out_path} ({canvas.size[0]/DPI:.2f}in x {canvas.size[1]/DPI:.2f}in)")


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
    build_vstack(paths, labels, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine per-island map figures into one vertical-stack figure.")
    parser.add_argument("--island", type=str, nargs="+", default=["a", "g", "o"])
    args = parser.parse_args()
    main(island_ids=args.island)
