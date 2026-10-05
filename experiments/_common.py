"""Plot helpers shared by the experiment scripts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "outputs/figures"


def annotated_heatmap(ax, mat, title, xlabels=None, ylabels=None, cmap="Blues", fmt="{:.2f}"):
    mat = np.asarray(mat, dtype=float)
    finite = np.where(np.isfinite(mat), mat, np.nan)
    ax.imshow(np.nan_to_num(finite, nan=np.nanmin(finite) if np.isfinite(finite).any() else 0), cmap=cmap)
    lo, hi = (np.nanmin(finite), np.nanmax(finite)) if np.isfinite(finite).any() else (0, 1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            frac = 0.0 if not np.isfinite(v) or hi == lo else (v - lo) / (hi - lo)
            ax.text(j, i, "-inf" if np.isneginf(v) else fmt.format(v), ha="center", va="center", fontsize=9,
                    color="white" if frac > 0.6 else "black")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(mat.shape[1])); ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels(xlabels if xlabels is not None else range(1, mat.shape[1] + 1), fontsize=8)
    ax.set_yticklabels(ylabels if ylabels is not None else range(1, mat.shape[0] + 1), fontsize=8)


def text_figure(text, path, title=None):
    """Render monospace text as an image (handy as a 'terminal screenshot')."""
    lines = text.splitlines()
    fig = plt.figure(figsize=(10, 0.28 * len(lines) + 0.8))
    fig.text(0.02, 0.98, (title + "\n\n" if title else "") + text, family="monospace", fontsize=9, va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
