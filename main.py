"""KenKen End-to-End Pipeline: Image → Vision → Solver → Visualization.

Usage
-----
    # Solve a photograph and save the annotated result:
    python main.py prueba4.jpg

    # Save to a custom path and display the result in a window:
    python main.py prueba4.jpg -o resultado.png --show

    # Use a custom model checkpoint:
    python main.py prueba4.jpg --model vision/glyph_cnn.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kenken",
        description="Solve a KenKen puzzle photograph End-to-End.",
    )
    parser.add_argument(
        "image",
        help="Path to the input image (.jpg or .png).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="PATH",
        help=(
            "Where to save the annotated output image. "
            "Defaults to 'solved_<image-stem>.png' in the current directory."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to the CNN model checkpoint (default: vision/glyph_cnn.pt).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the solved board in a matplotlib window after saving.",
    )
    args = parser.parse_args()

    # ── Ensure project root is on sys.path when run from any directory ────────
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from vision.model import DEFAULT_MODEL, load_model, parse_image_full
    from solver import solve
    from viz.renderer import draw_solution

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    output_path = Path(args.output) if args.output else Path(f"solved_{image_path.stem}.png")

    # ── Phase 1: Vision ───────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  KenKen Solver - {image_path.name}")
    print("=" * 60)

    t_start = time.perf_counter()

    print("[Fase 1] Cargando modelo CNN…")
    model = load_model(model_path)

    print("[Fase 1] Procesando imagen con el pipeline de visión…")
    result = parse_image_full(image_path, model)

    t_vision = time.perf_counter() - t_start

    n          = result["n"]
    warp       = result["warp"]
    xs         = result["xs"]
    ys         = result["ys"]
    cages_dict = result["cages"]

    print(
        f"[Fase 1] OK - {result['filename']} -> tablero {n}x{n}, "
        f"{len(cages_dict)} jaulas detectadas  ({t_vision * 1000:.1f} ms)"
    )

    # ── Phase 2: Solver ───────────────────────────────────────────────────────
    print("[Fase 2] Resolviendo con OR-Tools CP-SAT…")

    # convert cage dicts to the tuple format expected by solve()
    cage_tuples = [(c["target"], c["op"], c["cells"]) for c in cages_dict]

    t1 = time.perf_counter()
    solution = solve(n, cage_tuples)
    t_solver = time.perf_counter() - t1

    print(f"[Fase 2] OK - solución hallada ({t_solver * 1000:.3f} ms)")
    for row in solution:
        print("         " + "  ".join(str(v) for v in row))

    # ── Phase 3: Visualization ────────────────────────────────────────────────
    print("[Fase 3] Renderizando solución sobre el tablero…")

    t2 = time.perf_counter()
    canvas = draw_solution(warp, n, xs, ys, solution, cages=cages_dict)
    t_viz = time.perf_counter() - t2

    cv2.imwrite(str(output_path), canvas)

    t_total = time.perf_counter() - t_start
    print(f"[Fase 3] OK - imagen guardada en: {output_path}  ({t_viz * 1000:.1f} ms)")
    print("-" * 60)
    print(f"  Tiempo total End-to-End: {t_total * 1000:.1f} ms")
    print("=" * 60)

    # ── Optional display ──────────────────────────────────────────────────────
    if args.show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 9))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title(f"KenKen {n}x{n} - Solución", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
