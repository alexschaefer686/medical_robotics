"""Figure: boundary recognition in a vessel mask (features.boundary_nodes) on a 6 x 6 example.

The mask is a disc of radius 4 px centred on the lower-left image corner, so a quarter circle
of vessel (white, 1) shows in the image. boundary_nodes() puts a node at every inner pixel
corner and marks it as boundary when the 2 x 2 pixel block around it is mixed.

(a) the mask array and its nodes, with two 2 x 2 blocks marked; on the right the two blocks
zoomed in: a boundary node and a node that is not on the boundary.

python boundary_recognition.py    writes boundary_recognition.png next to this script
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from view_transition_rl.visual_servoing.features import boundary_nodes

# Times for the report; Times-like fallbacks where Times New Roman is not installed (e.g. Linux).
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral"]

N, RADIUS = 6, 4.0  # mask size and circle radius, px
BOUNDARY_NODE, INTERIOR_NODE = (2, 0), (3, 3)  # zoomed nodes (row, col); block = pixels rows i..i+1, cols j..j+1

BOUNDARY = "#e34948"  # boundary nodes: reference palette slot 8 (red)
BLOCK_1, BLOCK_2 = "#1baf7a", "#2a78d6"  # frames of the zoomed blocks: slots 3 (aqua) and 1 (blue)
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#8c8c8c"
ON_WHITE, ON_BLACK = INK, "#d9d9d9"  # text on vessel / background pixels


def quarter_disc_mask(n=N, radius=RADIUS):
    """(n, n) uint8 mask, 255 = vessel: pixels whose centre lies within radius of the lower-left corner."""
    rows, cols = np.mgrid[0:n, 0:n]
    return (((cols + 0.5) ** 2 + (n - (rows + 0.5)) ** 2 <= radius ** 2) * 255).astype(np.uint8)


def draw_pixels(ax, vessel):
    """Pixel (r, c) covers x in [c, c + 1], y in [r, r + 1], rows downwards."""
    n_rows, n_cols = vessel.shape
    ax.imshow(vessel, cmap="gray", vmin=0, vmax=1, extent=(0, n_cols, n_rows, 0), interpolation="nearest")
    for k in range(n_rows + 1):
        ax.plot([0, n_cols], [k, k], color=GRID, lw=0.8)
    for k in range(n_cols + 1):
        ax.plot([k, k], [0, n_rows], color=GRID, lw=0.8)


def draw_mask(ax, vessel, boundary):
    """The mask with its pixel values, the nodes at the inner pixel corners and the two zoomed blocks."""
    draw_pixels(ax, vessel)
    for (r, c), v in np.ndenumerate(vessel):
        ax.text(c + 0.5, r + 0.5, str(int(v)), ha="center", va="center", fontsize=18,
                color=ON_WHITE if v else ON_BLACK)
    rows, cols = np.indices(boundary.shape).reshape(2, -1)  # node (i, j) at x = j + 1, y = i + 1
    on = boundary[rows, cols]
    ax.scatter(cols[~on] + 1, rows[~on] + 1, s=30, facecolor="white", edgecolor=INK, lw=0.9, zorder=6)
    ax.scatter(cols[on] + 1, rows[on] + 1, s=70, facecolor=BOUNDARY, edgecolor=INK, lw=0.9, zorder=7)  # above the frames
    for (i, j), color in [(BOUNDARY_NODE, BLOCK_1), (INTERIOR_NODE, BLOCK_2)]:
        ax.add_patch(Rectangle((j, i), 2, 2, fill=False, edgecolor=color, lw=3, clip_on=False, zorder=5))

    n = vessel.shape[0]
    ax.set_xlim(-0.15, n + 0.15)
    ax.set_ylim(n + 0.15, -0.15)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5, [str(k) for k in range(n)])
    ax.set_yticks(np.arange(n) + 0.5, [str(k) for k in range(n)])
    ax.tick_params(length=0, labelsize=17, labelcolor=INK_2)
    ax.set_xlabel("column (probe x)", fontsize=18, color=INK_2)
    ax.set_ylabel("row (probe z)", fontsize=18, color=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"(a) {n} × {n} binary mask", fontsize=21, color=INK, loc="left")


def draw_block(ax, vessel, boundary, node, color):
    """The 2 x 2 pixel block around node with its pixels a, b, c, d; the node as boundary_nodes() marks it."""
    i, j = node
    block = vessel[i:i + 2, j:j + 2]
    is_boundary = bool(boundary[i, j])
    draw_pixels(ax, block)
    for (r, col), name in zip(np.ndindex(2, 2), "abcd"):
        v = block[r, col]
        ax.text(col + 0.5, r + 0.32, name, ha="center", va="center", fontsize=18, style="italic",
                color=ON_WHITE if v else ON_BLACK)
        ax.text(col + 0.5, r + 0.66, str(int(v)), ha="center", va="center", fontsize=26,
                color=ON_WHITE if v else ON_BLACK)
    ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, edgecolor=color, lw=3, clip_on=False, zorder=5))
    ax.scatter([1], [1], s=130, facecolor=BOUNDARY if is_boundary else "white", edgecolor=INK, lw=1.0, zorder=6)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(2.1, -0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Node ({i}, {j}) in (a) → {'boundary' if is_boundary else 'no boundary'}",
                 fontsize=21, color=INK, loc="center")  # the block is centred in its column, so under its title


def main():
    mask = quarter_disc_mask()
    vessel = (mask > 0).astype(int)
    boundary = boundary_nodes(mask)

    fig = plt.figure(figsize=(11.5, 6.6), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.55], hspace=0.35, wspace=0.15)  # room for the centred titles
    draw_mask(fig.add_subplot(grid[:, 0]), vessel, boundary)
    draw_block(fig.add_subplot(grid[0, 1]), vessel, boundary, BOUNDARY_NODE, BLOCK_1)
    draw_block(fig.add_subplot(grid[1, 1]), vessel, boundary, INTERIOR_NODE, BLOCK_2)

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
