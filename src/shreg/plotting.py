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
    COMPARISON_SIZE = (12, 6)  # 2 square subplots side by side


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


def _make_bounds_square(xmin: float, xmax: float, ymin: float, ymax: float) -> Tuple[float, float, float, float]:
    """Expand bounds to make them square, centered on the original bounds."""
    x_range = xmax - xmin
    y_range = ymax - ymin

    if x_range > y_range:
        # Expand y to match x
        diff = (x_range - y_range) / 2
        ymin -= diff
        ymax += diff
    else:
        # Expand x to match y
        diff = (y_range - x_range) / 2
        xmin -= diff
        xmax += diff

    return xmin, xmax, ymin, ymax


def _setup_axes(
    ax: plt.Axes,
    title: str | None = None,
    show_grid: bool = True,
    show_ticks: bool = True,
) -> None:
    """Configure axes with consistent styling."""
    ax.set_aspect("equal")
    if show_grid:
        ax.grid(True, alpha=Style.GRID_ALPHA)
    else:
        ax.grid(False)
    if title:
        ax.set_title(title)
    if not show_ticks:
        # Hide tick labels and tick marks, but keep tick positions for grid
        ax.tick_params(
            left=False, bottom=False,
            labelleft=False, labelbottom=False
        )


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
    title: str | None = "Before vs After",
    show: bool = True,
    save_path: str | None = None,
    show_titles: bool = True,
    show_ticks: bool = True,
    show_grid: bool = True,
) -> plt.Figure | None:
    """Plot before and after segments side by side with consistent axes.

    Args:
        before: Segments before transformation
        after: Segments after transformation
        title: Plot title (None to disable main title)
        show: Whether to call plt.show()
        save_path: Optional path to save the figure
        show_titles: Whether to show "Before"/"After" subplot titles
        show_ticks: Whether to show axis ticks and labels
        show_grid: Whether to show grid lines

    Returns:
        Figure object if show=False, None otherwise
    """
    if not ENABLED and save_path is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Style.COMPARISON_SIZE)

    # Calculate shared bounds from both before and after, make square
    all_segs = before + after
    xmin, xmax, ymin, ymax = _get_segments_bounds(all_segs)
    xmin, xmax, ymin, ymax = _make_bounds_square(xmin, xmax, ymin, ymax)

    for seg in before:
        ax1.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color=Style.SEGMENT_COLOR,
            linewidth=Style.SEGMENT_WIDTH,
        )
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    _setup_axes(ax1, "Before" if show_titles else None, show_grid=show_grid, show_ticks=show_ticks)

    for seg in after:
        ax2.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color=Style.AFTER_COLOR,
            linewidth=Style.SEGMENT_WIDTH,
        )
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    _setup_axes(ax2, "After" if show_titles else None, show_grid=show_grid, show_ticks=show_ticks)

    if title and show_titles:
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
    title: str | None = "Contour Regularization",
    show: bool = True,
    save_path: str | None = None,
    show_titles: bool = True,
    show_ticks: bool = True,
    show_grid: bool = True,
    show_point_counts: bool = True,
) -> plt.Figure | None:
    """Plot before and after contours side by side with consistent axes.

    Args:
        before_points: Points forming the original contour
        after_points: Points forming the regularized contour
        title: Plot title (None to disable main title)
        show: Whether to call plt.show()
        save_path: Optional path to save the figure
        show_titles: Whether to show "Before"/"After" subplot titles
        show_ticks: Whether to show axis ticks and labels
        show_grid: Whether to show grid lines
        show_point_counts: Whether to show point counts in subtitles

    Returns:
        Figure object if show=False, None otherwise
    """
    if not ENABLED and save_path is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Style.COMPARISON_SIZE)

    # Calculate shared bounds, make square
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
    xmin, xmax, ymin, ymax = _make_bounds_square(xmin, xmax, ymin, ymax)

    # Plot before
    before_closed = list(before_points) + [before_points[0]]
    bx, by = zip(*before_closed)
    ax1.plot(bx, by, color=Style.SEGMENT_COLOR, linewidth=Style.SEGMENT_WIDTH)
    ax1.scatter([p[0] for p in before_points], [p[1] for p in before_points],
                color=Style.SEGMENT_COLOR, s=20, zorder=5)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    if show_titles:
        before_title = f"Before ({len(before_points)} points)" if show_point_counts else "Before"
    else:
        before_title = None
    _setup_axes(ax1, before_title, show_grid=show_grid, show_ticks=show_ticks)

    # Plot after
    after_closed = list(after_points) + [after_points[0]]
    ax, ay = zip(*after_closed)
    ax2.plot(ax, ay, color=Style.AFTER_COLOR, linewidth=Style.SEGMENT_WIDTH)
    ax2.scatter([p[0] for p in after_points], [p[1] for p in after_points],
                color=Style.AFTER_COLOR, s=20, zorder=5)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    if show_titles:
        after_title = f"After ({len(after_points)} points)" if show_point_counts else "After"
    else:
        after_title = None
    _setup_axes(ax2, after_title, show_grid=show_grid, show_ticks=show_ticks)

    if title and show_titles:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    if show and ENABLED:
        plt.show()
        return None
    return fig


def metric_comparison(
    before: List[Segment],
    after: List[Segment],
    title: str | None = "Metric Regularization",
    show: bool = True,
    save_path: str | None = None,
    show_titles: bool = True,
    show_ticks: bool = True,
    show_grid: bool = True,
    show_lengths: bool = True,
    length_precision: int = 2,
) -> plt.Figure | None:
    """Plot before and after segments with length annotations.

    Specialized comparison for metric regularization that displays
    the length of each segment above its center.

    Args:
        before: Segments before transformation
        after: Segments after transformation
        title: Plot title (None to disable main title)
        show: Whether to call plt.show()
        save_path: Optional path to save the figure
        show_titles: Whether to show "Before"/"After" subplot titles
        show_ticks: Whether to show axis ticks and labels
        show_grid: Whether to show grid lines
        show_lengths: Whether to show segment length annotations
        length_precision: Decimal places for length display

    Returns:
        Figure object if show=False, None otherwise
    """
    if not ENABLED and save_path is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Style.COMPARISON_SIZE)

    # Calculate shared bounds from both before and after, make square
    all_segs = before + after
    xmin, xmax, ymin, ymax = _get_segments_bounds(all_segs, margin=0.15)
    xmin, xmax, ymin, ymax = _make_bounds_square(xmin, xmax, ymin, ymax)

    # Calculate text height offset (0.25 * approximate text height in data coords)
    data_range = max(xmax - xmin, ymax - ymin)
    text_offset = 0.25 * (data_range * 0.03)  # ~3% of range is approx text height

    def draw_segments_with_lengths(ax, segs, color):
        for seg in segs:
            ax.plot(
                [seg[0], seg[2]],
                [seg[1], seg[3]],
                color=color,
                linewidth=Style.SEGMENT_WIDTH,
            )
            if show_lengths:
                # Calculate midpoint and length
                mid_x = (seg[0] + seg[2]) / 2
                mid_y = (seg[1] + seg[3]) / 2
                seg_length = geometry.length(seg)

                # Position text just above the midpoint (offset by 0.25 * text height)
                ax.text(
                    mid_x, mid_y + text_offset,
                    f"{seg_length:.{length_precision}f}",
                    fontsize=8,
                    ha="center",
                    va="bottom",  # Text bottom edge at offset position
                    color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85),
                )

    draw_segments_with_lengths(ax1, before, Style.SEGMENT_COLOR)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    _setup_axes(ax1, "Before" if show_titles else None, show_grid=show_grid, show_ticks=show_ticks)

    draw_segments_with_lengths(ax2, after, Style.AFTER_COLOR)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    _setup_axes(ax2, "After" if show_titles else None, show_grid=show_grid, show_ticks=show_ticks)

    if title and show_titles:
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
