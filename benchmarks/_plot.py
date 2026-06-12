"""Small helper to keep plot code out of each benchmark."""

from __future__ import annotations

import os
from typing import Iterable, Sequence


def save_line_plot(
    xs: Sequence[float],
    series: dict[str, Sequence[float]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_box_plot(
    series: dict[str, Iterable[float]],
    *,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = list(series.keys())
    data = [list(series[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
