"""
Composite the three per-island map figures (viz/plot_island_maps.py's
output, run with --no-inset) into one 7x10in figure for the paper --
replaces three separate Figure 12/13/14 floats with one, per review
feedback asking for exactly that.

LAYOUT: Island A full-width on top, Islands B and C side-by-side below --
not a uniform grid. Each island's zoom extent (set by its own members'
and cut lines' real geographic spread, see plot_island_maps.py) has a
very different aspect ratio: Island A's cut spans a much larger north-
south than east-west distance (a tall/portrait panel), while Islands B
and C are the opposite (wide/landscape panels, B especially -- its two
members sit almost on the same latitude but far apart in longitude).
An earlier version of this script stacked all three at a common width,
which crushed the two landscape-shaped panels into thin, hard-to-read
strips; a later version split the page into a fixed 52/48 top/bottom
height share, which fit Island C well but left Island B -- the most
extreme aspect ratio of the three -- with large empty padding because
its natural width-filling height came nowhere close to the fixed row
height.

Row/column sizes are computed from the images' own aspect ratios instead
of a fixed or equal-width split. The top row gets exactly the height
Island A needs to fill the full page width. In the bottom row, B and C
do NOT get equal-width columns -- at equal widths, whichever of the two
is more extremely landscape-shaped (here, B) ends up padded with empty
space above/below because it needs much less height than C to fill the
same width. Instead, given a shared row height h, B's column is sized to
exactly h*aspect_B wide and C's to exactly h*aspect_C wide, and h itself
is solved so those two widths (plus the gap between them) exactly fill
the page width -- both panels then fill their own cell completely, with
zero padding in either, and the page naturally ends up with a wider
column for the more landscape-shaped island and a narrower one for the
other. If the resulting natural total height exceeds the 7x10in budget,
everything is scaled down uniformly to fit.

Each image is fit into its cell preserving its own aspect ratio (shrink
to fit, centered -- "letterboxed" with white on whichever axis doesn't
fill, never cropped or distorted -- these are geographic maps, so
distorting aspect ratio would misrepresent real distances).

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


def _fit_and_paste(canvas, img, cell_x0, cell_y0, cell_w, cell_h):
    """Shrink img to fit within (cell_w, cell_h) preserving its aspect ratio,
    and paste it centered in that cell -- letterboxed with white, never
    cropped or distorted."""
    scale = min(cell_w / img.width, cell_h / img.height)
    w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    resized = img.resize((w, h), Image.LANCZOS)
    x = cell_x0 + (cell_w - w) // 2
    y = cell_y0 + (cell_h - h) // 2
    canvas.paste(resized, (x, y))


def build_layout(paths, labels, out_path):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    if len(imgs) != 3:
        raise ValueError(f"This layout is hardcoded for exactly 3 islands (top + 2 below); got {len(imgs)}.")
    img_a, img_b, img_c = imgs

    canvas_w = int(FIG_W_IN * DPI)
    max_canvas_h = int(FIG_H_IN * DPI)
    aspect_a = img_a.width / img_a.height
    aspect_b = img_b.width / img_b.height
    aspect_c = img_c.width / img_c.height

    # Natural (no-padding) height each row needs. Top: Island A at full
    # page width. Bottom: solved so B's and C's columns -- each exactly
    # row_height * that image's own aspect ratio wide -- sum with the gap
    # to exactly canvas_w, so both fill their column completely with zero
    # padding (see module docstring for why this beats equal-width columns).
    natural_top_h = canvas_w / aspect_a
    natural_bottom_h = (canvas_w - GAP_PX) / (aspect_b + aspect_c)

    # Labels and the one inter-row gap are fixed pixel sizes, independent
    # of how much the image rows themselves get scaled -- only the IMAGE
    # portion of the height budget should be scaled down to fit, not the
    # overhead too (scaling overhead as if it were image content was a
    # real bug here: it understated how much vertical space the images
    # actually had left, so the scaled rows plus the untouched overhead
    # summed to MORE than max_canvas_h, clipping content at the page edge).
    overhead = 2 * LABEL_H_PX + GAP_PX
    available_image_h = max_canvas_h - overhead
    natural_image_h = natural_top_h + natural_bottom_h

    # If the natural row heights would exceed the available budget, scale
    # both down uniformly so they fit exactly; if there's slack instead,
    # leave it as bottom margin rather than stretched to fill, since
    # stretching would just reintroduce padding asymmetrically.
    scale = min(1.0, available_image_h / natural_image_h)
    top_h = int(natural_top_h * scale)
    bottom_h = int(natural_bottom_h * scale)
    width_b = int(bottom_h * aspect_b)
    width_c = int(bottom_h * aspect_c)
    canvas_h = top_h + bottom_h + overhead

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_label_font()

    def draw_label(label, x0, width, y):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x0 + (width - tw) / 2, y + 6), label, fill="black", font=font)

    y = 0
    draw_label(labels[0], 0, canvas_w, y)
    y += LABEL_H_PX
    _fit_and_paste(canvas, img_a, 0, y, canvas_w, top_h)
    y += top_h + GAP_PX

    draw_label(labels[1], 0, width_b, y)
    draw_label(labels[2], width_b + GAP_PX, width_c, y)
    y += LABEL_H_PX
    _fit_and_paste(canvas, img_b, 0, y, width_b, bottom_h)
    _fit_and_paste(canvas, img_c, width_b + GAP_PX, y, width_c, bottom_h)

    canvas.save(out_path, dpi=(DPI, DPI))
    logger.info(
        f"Saved {out_path} ({canvas_w/DPI:.2f}in x {canvas_h/DPI:.2f}in"
        f"{', scaled to fit 7x10in budget' if scale < 1.0 else ''})"
    )


def main(island_ids):
    if len(island_ids) != 3:
        raise ValueError("This layout (top + 2 below) expects exactly 3 islands.")
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
    parser = argparse.ArgumentParser(description="Combine the three per-island map figures into one 7x10in figure (A on top, B/C side-by-side below).")
    parser.add_argument("--island", type=str, nargs="+", default=["a", "g", "o"])
    args = parser.parse_args()
    main(island_ids=args.island)
