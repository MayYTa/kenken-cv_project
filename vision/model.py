"""Reusable KenKen vision pipeline.

Commands:
    python -m vision.model train
    python -m vision.model predict IMAGE --model vision/glyph_cnn.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASSES = list("0123456789") + ["+", "-", "*", "/"]
S = 2520
WALL_T = 0.5
DEFAULT_MODEL = Path(__file__).with_name("glyph_cnn.pt")
DEFAULT_PREDICTION = Path(__file__).with_name("pred.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, k: int = len(CLASSES)):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, k)
        self.dp = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return self.fc2(self.dp(F.relu(self.fc1(x.flatten(1)))))


def order_pts(points: np.ndarray) -> np.ndarray:
    total = points.sum(1)
    diff = np.diff(points, axis=1).ravel()
    return np.float32(
        [points[np.argmin(total)], points[np.argmin(diff)],
         points[np.argmax(total)], points[np.argmax(diff)]]
    )


def preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    threshold = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No board contour was found in the image.")
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(
        contour, 0.02 * cv2.arcLength(contour, True), True
    )
    if len(approx) == 4:
        source = order_pts(approx.reshape(4, 2).astype(np.float32))
    else:
        x, y, width, height = cv2.boundingRect(contour)
        source = np.float32(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
        )
    destination = np.float32(
        [[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]]
    )
    warp = cv2.warpPerspective(
        gray, cv2.getPerspectiveTransform(source, destination), (S, S)
    )
    _, ink = cv2.threshold(warp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return warp, ink


def _clusters(profile: np.ndarray, threshold: float) -> list[float]:
    indices = np.where(profile > threshold)[0]
    if len(indices) == 0:
        return []
    gap = max(8, round(8 * S / 600))
    return [group.mean() for group in np.split(
        indices, np.where(np.diff(indices) > gap)[0] + 1
    )]


def find_grid(warp: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    loose = (warp < 235).astype(np.uint8)
    xs = _clusters(loose.sum(0), 0.6 * S)
    ys = _clusters(loose.sum(1), 0.6 * S)
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Could not detect the board grid.")
    n = max(len(xs), len(ys)) - 1
    return n, np.linspace(xs[0], xs[-1], n + 1), np.linspace(ys[0], ys[-1], n + 1)


def edge_scores(
    ink: np.ndarray, n: int, xs: np.ndarray, ys: np.ndarray, hw: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if hw is None:
        hw = max(5, round(4 * S / 600))
    min_ink = max(3, round(2 * S / 600))
    vertical = np.zeros((n, n - 1))
    horizontal = np.zeros((n - 1, n))
    for r in range(n):
        y0 = int(ys[r] + 0.2 * (ys[r + 1] - ys[r]))
        y1 = int(ys[r] + 0.8 * (ys[r + 1] - ys[r]))
        for c in range(n - 1):
            x = int(round(xs[c + 1]))
            vertical[r, c] = (
                (ink[y0:y1, x - hw:x + hw + 1] > 0).sum(1) >= min_ink
            ).mean()
    for c in range(n):
        x0 = int(xs[c] + 0.2 * (xs[c + 1] - xs[c]))
        x1 = int(xs[c] + 0.8 * (xs[c + 1] - xs[c]))
        for r in range(n - 1):
            y = int(round(ys[r + 1]))
            horizontal[r, c] = (
                (ink[y - hw:y + hw + 1, x0:x1] > 0).sum(0) >= min_ink
            ).mean()
    return vertical, horizontal


def cage_cells(
    n: int, vertical: np.ndarray, horizontal: np.ndarray, threshold: float = WALL_T
) -> list[list[list[int]]]:
    seen = -np.ones((n, n), int)
    groups: list[list[list[int]]] = []
    for r0 in range(n):
        for c0 in range(n):
            if seen[r0, c0] >= 0:
                continue
            group_index = len(groups)
            seen[r0, c0] = group_index
            stack = [(r0, c0)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append([r, c])
                neighbours = []
                if c < n - 1 and vertical[r, c] < threshold:
                    neighbours.append((r, c + 1))
                if c > 0 and vertical[r, c - 1] < threshold:
                    neighbours.append((r, c - 1))
                if r < n - 1 and horizontal[r, c] < threshold:
                    neighbours.append((r + 1, c))
                if r > 0 and horizontal[r - 1, c] < threshold:
                    neighbours.append((r - 1, c))
                for rr, cc in neighbours:
                    if seen[rr, cc] < 0:
                        seen[rr, cc] = group_index
                        stack.append((rr, cc))
            groups.append(sorted(cells))
    return groups


def norm_glyph(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.where(mask)
    if len(yy) == 0:
        raise ValueError("Cannot normalize an empty glyph.")
    glyph = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1].astype(np.uint8) * 255
    height, width = glyph.shape
    side = max(height, width)
    square = np.zeros((side, side), np.uint8)
    square[(side - height) // 2:(side - height) // 2 + height,
           (side - width) // 2:(side - width) // 2 + width] = glyph
    square = cv2.copyMakeBorder(
        square, side // 8, side // 8, side // 8, side // 8,
        cv2.BORDER_CONSTANT, value=0
    )
    return cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)


def segment_label(ink: np.ndarray, xs: np.ndarray, ys: np.ndarray, r: int, c: int) -> list[np.ndarray]:
    x0, x1 = int(xs[c]) + 8, int(xs[c + 1]) - 4
    y0, y1 = int(ys[r]) + 8, int(ys[r] + 0.6 * (ys[r + 1] - ys[r]))
    crop = ink[y0:y1, x0:x1]
    number, labels, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
    components = []
    for i in range(1, number):
        x, y, width, height, area = stats[i]
        if area < 6 or x == 0 or y == 0 or x + width >= crop.shape[1] or y + height >= crop.shape[0]:
            continue
        components.append((x, x + width, i))
    components.sort()
    widths = [xb - xa for xa, xb, _ in components]
    typical = float(np.median(widths)) if len(widths) >= 2 else 0
    parts = []
    for xa, xb, component in components:
        mask = np.isin(labels, component)
        projection = mask[:, xa:xb].sum(0)
        if typical and xb - xa > 1.6 * typical:
            lo = max(1, len(projection) // 4)
            hi = min(len(projection) - 1, 3 * len(projection) // 4)
            cut = lo + int(np.argmin(projection[lo:hi]))
            if projection[cut] <= max(1, 0.15 * projection.max()):
                left = np.zeros_like(mask)
                right = np.zeros_like(mask)
                left[:, :xa + cut] = mask[:, :xa + cut]
                right[:, xa + cut:] = mask[:, xa + cut:]
                parts.extend([(xa, xa + cut, left), (xa + cut, xb, right)])
                continue
        parts.append((xa, xb, mask))
    groups: list[dict[str, Any]] = []
    for xa, xb, mask in parts:
        if groups and xa < groups[-1]["x1"]:
            groups[-1]["x1"] = max(groups[-1]["x1"], xb)
            groups[-1]["masks"].append(mask)
        else:
            groups.append({"x1": xb, "masks": [mask]})
    return [norm_glyph(np.logical_or.reduce(group["masks"])) for group in groups]


def decode(chars: list[str]) -> dict[str, Any] | None:
    if chars and chars[-1] in "+-*/":
        op, digits = chars[-1], chars[:-1]
    else:
        op, digits = "=", chars
    if not digits or any(not digit.isdigit() for digit in digits):
        return None
    return {"op": op, "target": int("".join(digits))}


@torch.no_grad()
def classify(model: Net, glyphs: list[np.ndarray]) -> list[str]:
    if not glyphs:
        return []
    values = torch.tensor(np.stack(glyphs)).float().div(255).unsqueeze(1).to(DEVICE)
    return [CLASSES[i] for i in model(values).argmax(1).tolist()]


def load_model(path: str | os.PathLike[str] = DEFAULT_MODEL) -> Net:
    model = Net().to(DEVICE)
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model file not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_image(path: str | os.PathLike[str], model: Net) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    warp, ink = preprocess(image)
    n, xs, ys = find_grid(warp)
    vertical, horizontal = edge_scores(ink, n, xs, ys)
    cages = []
    for cells in cage_cells(n, vertical, horizontal):
        r, c = min(map(tuple, cells))
        clue = decode(classify(model, segment_label(ink, xs, ys, r, c)))
        if clue is None:
            raise ValueError(f"Could not recognize clue at {(r, c)} in {path}.")
        cages.append({**clue, "cells": cells})
    return {"n": n, "filename": Path(path).name, "cages": cages}


def parse_image_full(path: str | os.PathLike[str], model: Net) -> dict[str, Any]:
    """Parse the image and return all intermediate data for visualization.
    
    Returns a dictionary containing:
      - 'n': grid size
      - 'filename': name of the image file
      - 'cages': list of parsed cages
      - 'warp': normalized board image (S x S)
      - 'xs', 'ys': grid line coordinates
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    warp, ink = preprocess(image)
    n, xs, ys = find_grid(warp)
    vertical, horizontal = edge_scores(ink, n, xs, ys)
    cages = []
    for cells in cage_cells(n, vertical, horizontal):
        r, c = min(map(tuple, cells))
        clue = decode(classify(model, segment_label(ink, xs, ys, r, c)))
        if clue is None:
            raise ValueError(f"Could not recognize clue at {(r, c)} in {path}.")
        cages.append({**clue, "cells": cells})
    return {
        "n": n,
        "filename": Path(path).name,
        "cages": cages,
        "warp": warp,
        "xs": xs,
        "ys": ys
    }


def export_json(paths: list[str], model: Net, out: str | os.PathLike[str]) -> str:
    paths = [str(path) for path in paths]
    if not paths:
        raise ValueError("No input images were provided.")
    boards = [parse_image(path, model) for path in paths]
    sizes = {board["n"] for board in boards}
    if len(sizes) != 1:
        raise ValueError("All images in one export must have the same grid size.")
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"n": boards[0]["n"], "images": boards}, indent=2),
        encoding="utf-8"
    )
    return str(output)


def _load_training_items(root: Path) -> list[dict[str, Any]]:
    items = []
    for json_path in sorted(root.glob("gt/*x*/*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        tag = json_path.parent.name
        for image in data.get("images", []):
            image_path = root / "images" / tag / image["filename"]
            if image_path.exists():
                items.append({"path": str(image_path), "n": data["n"], "cages": image["cages"]})
    return items


def _label_chars(cage: dict[str, Any]) -> list[str]:
    chars = list(str(cage["target"]))
    if cage["op"] != "=":
        chars.append(cage["op"])
    return chars


def _extract_samples(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    labels: list[int] = []
    for item in items:
        image = cv2.imread(item["path"])
        if image is None:
            continue
        warp, ink = preprocess(image)
        n, xs, ys = find_grid(warp)
        if n != item["n"]:
            continue
        for cage in item["cages"]:
            r, c = min(map(tuple, cage["cells"]))
            glyphs = segment_label(ink, xs, ys, r, c)
            chars = _label_chars(cage)
            if len(glyphs) == len(chars):
                samples.extend(glyphs)
                labels.extend(CLASSES.index(char) for char in chars)
    if not samples:
        raise ValueError("No glyphs could be extracted from the labeled dataset.")
    return np.stack(samples), np.asarray(labels, dtype=np.int64)


def _augment_glyph(glyph: np.ndarray) -> np.ndarray:
    """Create a small realistic variation of a normalized glyph."""
    angle = random.uniform(-7.0, 7.0)
    scale = random.uniform(0.94, 1.06)
    matrix = cv2.getRotationMatrix2D((16, 16), angle, scale)
    result = cv2.warpAffine(
        glyph,
        matrix,
        (32, 32),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if random.random() < 0.35:
        kernel = np.ones((2, 2), np.uint8)
        result = (
            cv2.dilate(result, kernel, iterations=1)
            if random.random() < 0.5
            else cv2.erode(result, kernel, iterations=1)
        )
    return result


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    loss: float,
) -> dict[str, Any]:
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        confusion[actual, predicted] += 1
    support = confusion.sum(axis=1)
    true_positive = np.diag(confusion)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    top_k = min(3, probabilities.shape[1])
    ranked_indices = np.argsort(probabilities, axis=1)[:, ::-1]
    top_indices = ranked_indices[:, :top_k]
    top_k_accuracy = float(np.any(top_indices == labels[:, None], axis=1).mean())
    confidence = probabilities[np.arange(len(labels)), predictions]
    errors = [
        {
            "index": int(index),
            "actual": CLASSES[actual],
            "predicted": CLASSES[predicted],
            "confidence": float(confidence[index]),
            "top_3": [CLASSES[i] for i in ranked_indices[index, :top_k]],
        }
        for index, (actual, predicted) in enumerate(zip(labels, predictions))
        if actual != predicted
    ]
    return {
        "samples": int(len(labels)),
        "loss": float(loss),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(np.mean(recall[support > 0])) if np.any(support) else 0.0,
        "top_3_accuracy": top_k_accuracy,
        "macro_precision": float(np.mean(precision[support > 0])) if np.any(support) else 0.0,
        "macro_recall": float(np.mean(recall[support > 0])) if np.any(support) else 0.0,
        "macro_f1": float(np.mean(f1[support > 0])) if np.any(support) else 0.0,
        "weighted_precision": float(np.average(precision, weights=support)) if support.sum() else 0.0,
        "weighted_recall": float(np.average(recall, weights=support)) if support.sum() else 0.0,
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
        "mean_confidence": float(confidence.mean()),
        "mean_correct_confidence": float(confidence[labels == predictions].mean())
        if np.any(labels == predictions) else 0.0,
        "mean_incorrect_confidence": float(confidence[labels != predictions].mean())
        if np.any(labels != predictions) else 0.0,
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            char: {
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, char in enumerate(CLASSES)
        },
        "labels": [CLASSES[index] for index in labels],
        "predictions": [CLASSES[index] for index in predictions],
        "confidences": confidence.tolist(),
        "errors": errors,
    }


@torch.no_grad()
def evaluate_model(
    model: Net,
    root: str | os.PathLike[str],
    validation_fraction: float = 0.2,
    seed: int = 42,
    split: str = "validation",
    plot: bool = False,
) -> dict[str, Any]:
    """Evaluate glyph recognition and return machine-readable metrics.

    Boards, rather than individual glyphs, are split so glyphs from one image
    cannot appear in both training and validation metrics. ``split`` can be
    ``"all"``, ``"train"`` or ``"validation"``.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if split not in {"all", "train", "validation"}:
        raise ValueError("split must be 'all', 'train' or 'validation'.")
    items = _load_training_items(Path(root))
    if not items:
        raise ValueError(f"No labeled training boards found under {root}.")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cut = round(len(shuffled) * (1 - validation_fraction))
    selected = shuffled if split == "all" else (
        shuffled[:cut] if split == "train" else shuffled[cut:]
    )
    if not selected:
        raise ValueError(f"The {split} split contains no boards.")
    images, labels = _extract_samples(selected)
    values = torch.tensor(images).float().div(255).unsqueeze(1).to(DEVICE)
    targets = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    model.eval()
    logits = model(values)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    loss = F.cross_entropy(logits, targets).item()
    metrics = _classification_metrics(labels, predictions, probabilities, loss)
    metrics.update({
        "split": split,
        "boards": len(selected),
        "validation_fraction": validation_fraction,
        "seed": seed,
    })
    if plot:
        import matplotlib.pyplot as plt

        matrix = np.asarray(metrics["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(9, 7))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=range(len(CLASSES)),
            yticks=range(len(CLASSES)),
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            xlabel="Predicción",
            ylabel="Etiqueta real",
            title=f"Matriz de confusión ({split})",
        )
        for row in range(len(CLASSES)):
            for column in range(len(CLASSES)):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
        figure.tight_layout()
        metrics["confusion_figure"] = figure
    return metrics


def train_model(root: str | os.PathLike[str], output: str | os.PathLike[str] = DEFAULT_MODEL, epochs: int = 60) -> Net:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    items = _load_training_items(Path(root))
    if not items:
        raise ValueError(f"No labeled training boards found under {root}.")
    samples, labels = _extract_samples(items)
    model = Net().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    x = torch.tensor(samples).float().div(255).unsqueeze(1).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights /= weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    operator_indices = {CLASSES.index("+"), CLASSES.index("/")}
    for epoch in range(epochs):
        model.train()
        augmented = np.stack([
            _augment_glyph(glyph) if label in operator_indices else glyph
            for glyph, label in zip(samples, labels)
        ])
        x_epoch = torch.tensor(augmented).float().div(255).unsqueeze(1).to(DEVICE)
        logits = model(x_epoch)
        loss = F.cross_entropy(logits, y, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        accuracy = (logits.argmax(1) == y).float().mean().item()
        print(f"epoch {epoch + 1:3d} | train loss {loss.item():.4f} | train acc {accuracy:.4f}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    return model


def get_or_train_model(
    model_path: str | os.PathLike[str] = DEFAULT_MODEL,
    train_root: str | os.PathLike[str] = "dataset",
    epochs: int = 60,
) -> Net:
    if Path(model_path).exists():
        return load_model(model_path)
    print(f"Model not found at {model_path}; training it now.")
    return train_model(train_root, model_path, epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description="KenKen glyph model and image exporter.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--dataset", default="dataset")
    train.add_argument("--model", default=str(DEFAULT_MODEL))
    train.add_argument("--epochs", type=int, default=60)
    predict = subparsers.add_parser("predict")
    predict.add_argument("image")
    predict.add_argument("--model", default=str(DEFAULT_MODEL))
    predict.add_argument("--dataset", default="dataset")
    predict.add_argument("--epochs", type=int, default=60)
    predict.add_argument("--out", default=str(DEFAULT_PREDICTION))
    args = parser.parse_args()
    if args.command == "train":
        train_model(args.dataset, args.model, args.epochs)
    else:
        model = get_or_train_model(args.model, args.dataset, args.epochs)
        print(f"Exported: {export_json([args.image], model, args.out)}")


if __name__ == "__main__":
    main()
