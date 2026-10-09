"""Visualization module for the KenKen End-to-End pipeline.

Provides draw_solution(), which overlays solver output onto the canonical
2520×2520 warp canvas produced by vision.model.preprocess().
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_solution(
    warp: np.ndarray,
    n: int,
    xs: np.ndarray,
    ys: np.ndarray,
    solution: list[list[int]],
    cages: list[dict] | None = None,
) -> np.ndarray:
    """Overlay the solver's numeric solution on the canonical warp canvas.

    Parameters
    ----------
    warp:
        Grayscale canonical canvas of shape (S, S) — typically 2520×2520 —
        produced by vision.model.preprocess().
    n:
        Grid dimension (e.g. 6 for a 6×6 board).
    xs:
        Array of n+1 X-coordinates for the vertical grid lines, produced by
        vision.model.find_grid(). xs[c] and xs[c+1] are the left and right
        borders of column c.
    ys:
        Array of n+1 Y-coordinates for the horizontal grid lines.
        ys[r] and ys[r+1] are the top and bottom borders of row r.
    solution:
        2-D list returned by solver.solve() where solution[r][c] is the
        integer value placed in cell (r, c).
    cages:
        Optional list of cage dicts (as returned by vision.model.parse_image_full).
        Each dict has keys 'target', 'op', and 'cells'. When provided, the
        clue label (e.g. "12*", "3=") is drawn in the top-left corner of each
        cage in a contrasting smaller font.

    Returns
    -------
    np.ndarray
        BGR image of the same spatial size as *warp* with the solution numbers
        and optional cage clues drawn on top of the board.
    """
    # ── 1. Convert grayscale canvas to BGR so we can draw in colour ──────────
    canvas = cv2.cvtColor(warp, cv2.COLOR_GRAY2BGR)

    # ── 2. Derive proportional typography parameters from cell size ──────────
    # cell_size is the smaller dimension so the font fits regardless of aspect.
    cell_w = float(xs[-1] - xs[0]) / n
    cell_h = float(ys[-1] - ys[0]) / n
    cell_size = min(cell_w, cell_h)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale grows with cell size so numbers fill the cell for any n.
    # For n=6  (cell≈420 px) → scale≈5.25
    # For n=9  (cell≈280 px) → scale≈3.50
    font_scale = cell_size / 80.0
    # thickness is also proportional, minimum 2 px for sharp rendering.
    thickness = max(2, int(cell_size / 100))
    # Dark green on white/grey background — high contrast, distinct from clues.
    color_solution = (0, 140, 0)  # BGR

    # ── 3. Draw each solution digit, centred in its cell ─────────────────────
    for r in range(n):
        for c in range(n):
            text = str(solution[r][c])

            # Cell centre in canvas pixel coordinates.
            x_center = (xs[c] + xs[c + 1]) / 2.0
            y_center = (ys[r] + ys[r + 1]) / 2.0

            # Measure the rendered text so we can compute the anchor point.
            # cv2.getTextSize returns (width, height) of the tight bounding box
            # and the distance from the baseline to the bottom of the box.
            (text_w, text_h), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # cv2.putText places the text with its BOTTOM-LEFT corner at
            # (x_org, y_org).  To visually centre:
            #   x_org = x_center − text_w / 2   (shift left by half the width)
            #   y_org = y_center + text_h / 2   (shift down by half the height)
            x_org = int(x_center - text_w / 2)
            y_org = int(y_center + text_h / 2)

            cv2.putText(
                canvas, text, (x_org, y_org),
                font, font_scale, color_solution, thickness, cv2.LINE_AA,
            )

    # ── 4. Optionally overlay the cage clue labels ────────────────────────────
    if cages is not None:
        clue_scale = font_scale * 0.35
        clue_thickness = max(1, thickness // 2)
        color_clue = (180, 0, 0)  # Dark blue BGR — visually distinct from green

        for cage in cages:
            # Cage may be a dict (from parse_image_full) or a tuple (target, op, cells).
            if isinstance(cage, dict):
                target = cage["target"]
                op = cage["op"]
                cells = cage["cells"]
            else:
                target, op, cells = cage[0], cage[1], cage[2]

            # The clue is always printed in the top-left cell of the cage.
            r_clue, c_clue = map(int, min(map(tuple, cells)))

            # Format: "12*", "3=", "2/", etc. Omit the "=" for singleton cells.
            clue_text = str(target) if op == "=" else f"{target}{op}"

            # Position: a small margin inside the top-left corner of the cell.
            margin = max(8, int(cell_size * 0.04))
            x_clue = int(xs[c_clue]) + margin
            # Simpler: just offset by ~20% of cell height from the top border.
            y_clue = int(ys[r_clue]) + int(cell_h * 0.22)

            cv2.putText(
                canvas, clue_text, (x_clue, y_clue),
                font, clue_scale, color_clue, clue_thickness, cv2.LINE_AA,
            )

    return canvas
