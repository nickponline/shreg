"""Plotting utilities for segment visualization.

Set ENABLED = False to disable all plotting globally.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import geometry
from .geometry import Segment

# Global flag to enable/disable all plotting
ENABLED = True


# Style configuration
class Style:
    SEGMENT_COLOR = "#2C3E50"
    SEGMENT_WIDTH = 1.5
    CONSTRAINT_SEGMENT_COLOR = "#7F8C8D"
    CONSTRAINT_SEGMENT_WIDTH = 0.3
    EDGE_COLOR = "#E74C3C"
    EDGE_WIDTH = 0.5
    DEBUG_EDGE_COLOR = "#3498DB"
    DEBUG_EDGE_WIDTH = 0.2
    POINT_COLOR = "#27AE60"
    AFTER_COLOR = "#27AE60"
    TEXT_SIZE = 8
    GRID_ALPHA = 0.3
    FIGURE_SIZE = (8, 8)
    COMPARISON_SIZE = (12, 5)


def _get_segments_bounds(segs: List[Segment], margin: float = 0.1) -> Tuple[float, float, float, float]:
    """Calculate bounding box for segments with margin."""
    if not segs:
        return 0, 1, 0, 1

    all_x = []
    all_y = []
    for seg in segs:
        all_x.extend([seg[0], seg[2]])
        all_y.extend([seg[1], seg[3]])

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    # Add margin
    dx = (xmax - xmin) * margin or 1
    dy = (ymax - ymin) * margin or 1

    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def _setup_axes(ax: plt.Axes, title: str | None = None) -> None:
    """Configure axes with consistent styling."""
    ax.set_aspect("equal")
    ax.grid(True, alpha=Style.GRID_ALPHA)
    if title:
        ax.set_title(title)


def segments(
    segs: List[Segment],
    title: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Plot a list of segments.

    Args:
        segs: List of segments to plot
        title: Optional title for the plot
        show: Whether to call plt.show()
        ax: Optional existing axes to plot on
        save_path: Optional path to save the figure

    Returns:
        Figure object if show=False, None otherwise
    """
    if not ENABLED and save_path is None:
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=Style.FIGURE_SIZE)
    else:
        fig = ax.get_figure()

    for seg in segs:
        ax.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color=Style.SEGMENT_COLOR,
            linewidth=Style.SEGMENT_WIDTH,
        )

    _setup_axes(ax, title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    if show and ENABLED:
        plt.show()
        return None
    return fig


def constraints(
    segs: List[Segment],
    edges: List[tuple[int, int]],
    error_func: Callable[[Segment, Segment], float],
    title: str = "Constraints",
    show: bool = True,
) -> plt.Figure | None:
    """Plot segments with constraint edges and error values.

    Args:
        segs: List of segments
        edges: List of edge tuples (i, j) indicating constraints
        error_func: Function to compute error between two segments
        title: Plot title
        show: Whether to call plt.show()

    Returns:
        Figure object if show=False, None otherwise
    """
    if not ENABLED:
        return None

    fig, ax = plt.subplots(figsize=Style.FIGURE_SIZE)

    # Draw segments
    for seg in segs:
        ax.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color=Style.CONSTRAINT_SEGMENT_COLOR,
            linewidth=Style.CONSTRAINT_SEGMENT_WIDTH,
        )

    # Draw constraint edges with error labels
    for a, b in edges:
        error = error_func(segs[a], segs[b])
        mid_a = geometry.midpoint(segs[a])
        mid_b = geometry.midpoint(segs[b])

        ax.plot(
            [mid_a[0], mid_b[0]],
            [mid_a[1], mid_b[1]],
            color=Style.EDGE_COLOR,
            linewidth=Style.EDGE_WIDTH,
        )
        ax.text(
            (mid_a[0] + mid_b[0]) / 2,
            (mid_a[1] + mid_b[1]) / 2,
            f"{error:.2f}",
            fontsize=Style.TEXT_SIZE,
            ha="center",
            va="center",
        )

    _setup_axes(ax, title)

    if show:
        plt.show()
        return None
    return fig


def delaunay_debug(
    segs: List[Segment],
    tripoints: np.ndarray,
    error_func: Callable[[Segment, Segment], float],
    title: str = "Delaunay Triangulation",
    show: bool = True,
) -> plt.Figure | None:
    """Plot segments with Delaunay triangulation edges for debugging.

    Args:
        segs: List of segments
        tripoints: Triangulation simplices array
        error_func: Function to compute error between two segments
        title: Plot title
        show: Whether to call plt.show()

    Returns:
        Figure object if show=False, None otherwise
    """
    if not ENABLED:
        return None

    fig, ax = plt.subplots(figsize=Style.FIGURE_SIZE)

    # Draw segments
    for seg in segs:
        ax.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color=Style.SEGMENT_COLOR,
            linewidth=Style.SEGMENT_WIDTH * 0.5,
        )

    # Draw triangulation edges with error labels
    for v1, v2, v3 in tripoints:
        mids = [geometry.midpoint(segs[v]) for v in (v1, v2, v3)]
        pairs = [(v1, v2, 0, 1), (v2, v3, 1, 2), (v3, v1, 2, 0)]

        for va, vb, ma, mb in pairs:
            error = error_func(segs[va], segs[vb])
            ax.plot(
                [mids[ma][0], mids[mb][0]],
                [mids[ma][1], mids[mb][1]],
                color=Style.DEBUG_EDGE_COLOR,
                linewidth=Style.DEBUG_EDGE_WIDTH,
            )
            ax.text(
                (mids[ma][0] + mids[mb][0]) / 2,
                (mids[ma][1] + mids[mb][1]) / 2,
                f"{error:.2f}",
                fontsize=Style.TEXT_SIZE,
                ha="center",
                va="center",
            )

    _setup_axes(ax, title)

    if show:
        plt.show()
        return None
    return fig


def points_and_segments(
    segs: List[Segment],
    points: List[tuple[float, float]],
    title: str | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Plot segments with overlaid points.

    Args:
        segs: List of segments
        points: List of (x, y) point coordinates
        title: Optional plot title
        show: Whether to call plt.show()

    Returns:
        Figure object if show=False, None otherwise
    """
    if not ENABLED:
        return None

    fig, ax = plt.subplots(figsize=Style.FIGURE_SIZE)

    # Draw segments
    for seg in segs:
        ax.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color=Style.SEGMENT_COLOR,
            linewidth=Style.SEGMENT_WIDTH * 0.5,
        )

    # Draw points
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, color=Style.POINT_COLOR, s=10)

    _setup_axes(ax, title)

    if show:
        plt.show()
        return None
    return fig


def comparison(
    before: List[Segment],
    after: List[Segment],
    title: str = "Before vs After",
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Plot before and after segments side by side with consistent axes.

    Args:
        before: Segments before transformation
        after: Segments after transformation
        title: Plot title
        show: Whether to call plt.show()
        save_path: Optional path to save the figure

    Returns:
        Figure object if show=False, None otherwise
    """
    if not ENABLED and save_path is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Style.COMPARISON_SIZE)

    # Calculate shared bounds from both before and after
    all_segs = before + after
    xmin, xmax, ymin, ymax = _get_segments_bounds(all_segs)

    for seg in before:
        ax1.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color=Style.SEGMENT_COLOR,
            linewidth=Style.SEGMENT_WIDTH,
        )
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    _setup_axes(ax1, "Before")

    for seg in after:
        ax2.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color=Style.AFTER_COLOR,
            linewidth=Style.SEGMENT_WIDTH,
        )
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    _setup_axes(ax2, "After")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    if show and ENABLED:
        plt.show()
        return None
    return fig


def contour_comparison(
    before_points: List[Tuple[float, float]],
    after_points: List[Tuple[float, float]],
    title: str = "Contour Regularization",
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Plot before and after contours side by side with consistent axes.

    Args:
        before_points: Points forming the original contour
        after_points: Points forming the regularized contour
        title: Plot title
        show: Whether to call plt.show()
        save_path: Optional path to save the figure

    Returns:
        Figure object if show=False, None otherwise
    """
    if not ENABLED and save_path is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Style.COMPARISON_SIZE)

    # Calculate shared bounds
    all_points = list(before_points) + list(after_points)
    all_x = [p[0] for p in all_points]
    all_y = [p[1] for p in all_points]
    margin = 0.1
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    dx = (xmax - xmin) * margin or 1
    dy = (ymax - ymin) * margin or 1
    xmin, xmax = xmin - dx, xmax + dx
    ymin, ymax = ymin - dy, ymax + dy

    # Plot before
    before_closed = list(before_points) + [before_points[0]]
    bx, by = zip(*before_closed)
    ax1.plot(bx, by, color=Style.SEGMENT_COLOR, linewidth=Style.SEGMENT_WIDTH)
    ax1.scatter([p[0] for p in before_points], [p[1] for p in before_points],
                color=Style.SEGMENT_COLOR, s=20, zorder=5)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    _setup_axes(ax1, f"Before ({len(before_points)} points)")

    # Plot after
    after_closed = list(after_points) + [after_points[0]]
    ax, ay = zip(*after_closed)
    ax2.plot(ax, ay, color=Style.AFTER_COLOR, linewidth=Style.SEGMENT_WIDTH)
    ax2.scatter([p[0] for p in after_points], [p[1] for p in after_points],
                color=Style.AFTER_COLOR, s=20, zorder=5)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    _setup_axes(ax2, f"After ({len(after_points)} points)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    if show and ENABLED:
        plt.show()
        return None
    return fig


def disable() -> None:
    """Disable all plotting."""
    global ENABLED
    ENABLED = False


def enable() -> None:
    """Enable all plotting."""
    global ENABLED
    ENABLED = True


class BeforeAfter:
    """Context manager for before/after segment visualization.

    Usage:
        with plotting.BeforeAfter("My Title") as plot:
            plot.before(segments)
            segments = transform(segments)
            plot.after(segments)
    """

    def __init__(self, title: str = "Before vs After", save_path: str | None = None):
        self.title = title
        self.save_path = save_path
        self._before: List[Segment] | None = None
        self._after: List[Segment] | None = None

    def __enter__(self) -> "BeforeAfter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not ENABLED and self.save_path is None:
            return
        if self._before is not None and self._after is not None:
            comparison(self._before, self._after, title=self.title, save_path=self.save_path)

    def before(self, segs: List[Segment]) -> None:
        """Record the 'before' state."""
        self._before = [s.copy() for s in segs]

    def after(self, segs: List[Segment]) -> None:
        """Record the 'after' state."""
        self._after = [s.copy() for s in segs]


def before_after(
    before_segs: List[Segment],
    after_segs: List[Segment],
    title: str = "Before vs After",
    save_path: str | None = None,
) -> None:
    """Simple function to plot before/after comparison.

    Args:
        before_segs: Segments before transformation
        after_segs: Segments after transformation
        title: Plot title
        save_path: Optional path to save the figure
    """
    if not ENABLED and save_path is None:
        return
    comparison(before_segs, after_segs, title=title, save_path=save_path)
