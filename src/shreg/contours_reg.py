"""Contour regularization following CGAL Shape Regularization.

Reference: https://doc.cgal.org/latest/Shape_regularization/index.html#title10

This module implements contour regularization which:
1. Rotates segments to align with principal directions (modulo 90 degrees)
2. Merges parallel segments that are close together
3. Computes intersection points to form a regularized closed contour
"""
from __future__ import annotations

import importlib.resources
from typing import List, Literal, Tuple

import numpy as np

from . import geometry
from . import plotting
from .geometry import Point, Segment


def segments_from_points(points: List[Tuple[float, float]]) -> List[Segment]:
    """Create segments from a list of points forming a closed contour."""
    segments = []
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        seg = np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float64)
        segments.append(seg)
    return segments


def segment_offset(s: Segment, t: Segment) -> float:
    """Calculate the perpendicular offset between two segments."""
    return abs(geometry.signed_distance_to_point(t, geometry.midpoint(s)))


def is_parallel(s: Segment, t: Segment, tolerance_degrees: float = 1.0) -> bool:
    """Check if two segments are parallel within a tolerance."""
    return abs(geometry.orientation(s) - geometry.orientation(t)) < tolerance_degrees


def make_segment_from_midpoint(
    midpoint: Point, orientation_deg: float, seg_length: float
) -> Segment:
    """Create a segment from midpoint, orientation (degrees), and length."""
    angle_rad = np.radians(orientation_deg)
    half = seg_length / 2
    x1 = midpoint[0] + np.cos(angle_rad) * half
    y1 = midpoint[1] + np.sin(angle_rad) * half
    x2 = midpoint[0] - np.cos(angle_rad) * half
    y2 = midpoint[1] - np.sin(angle_rad) * half
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def line_intersection(s: Segment, t: Segment) -> Tuple[float, float] | None:
    """Find intersection point of two lines (extended infinitely)."""
    x1, y1, x2, y2 = s
    x3, y3, x4, y4 = t

    xdiff = (x1 - x2, x3 - x4)
    ydiff = (y1 - y2, y3 - y4)

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if abs(div) < 1e-10:
        return None  # Lines are parallel

    d = (det((x1, y1), (x2, y2)), det((x3, y3), (x4, y4)))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def project_point_onto_line(
    point: Tuple[float, float], line: Segment
) -> Tuple[float, float]:
    """Project a point onto a line segment (extended infinitely)."""
    px, py = point
    x1, y1, x2, y2 = line

    line_vec = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return x1, y1

    unit_line = line_vec / line_len
    point_vec = np.array([px - x1, py - y1], dtype=np.float64)
    proj_length = np.dot(point_vec, unit_line)
    proj_point = np.array([x1, y1]) + proj_length * unit_line

    return float(proj_point[0]), float(proj_point[1])


def longest_segment(segments: List[Segment]) -> Segment:
    """Return the longest segment from a list."""
    return max(segments, key=geometry.length)


def regularize_contour(
    points: List[Tuple[float, float]],
    principle: Literal["longest", "axis", "cardinal"] = "longest",
    max_offset: float = 20.0,
    visualize: bool = False,
) -> np.ndarray:
    """Regularize a closed contour of points.

    This function:
    1. Creates segments from the input points
    2. Rotates each segment to align with principal directions (modulo 90 degrees)
    3. Merges parallel segments that are close together
    4. Computes intersection points to form a regularized closed contour

    Args:
        points: List of (x, y) coordinates forming a closed contour
        principle: How to determine principal directions:
            - "longest": Use the longest segment as reference
            - "axis": Use vertical axis (0 degrees) as reference
            - "cardinal": Use cardinal directions (0, 45, 90, 135 degrees)
        max_offset: Maximum offset for merging parallel segments
        visualize: Whether to show intermediate plots

    Returns:
        Array of regularized points forming a closed contour
    """
    # Create segments from points
    segments = segments_from_points(points)
    segments = [s for s in segments if geometry.length(s) > 0.0]

    if visualize:
        plotting.segments(segments, title="Input Segments")

    # Determine reference orientations
    longest_seg = longest_segment(segments)

    if principle == "longest":
        references = [longest_seg]
    elif principle == "axis":
        # Vertical reference
        ref_seg = make_segment_from_midpoint(
            np.array([0, 0]), 90.0, 1.0
        )
        references = [ref_seg]
    elif principle == "cardinal":
        # Cardinal directions: 0, 45, 90, 135 degrees
        references = [
            make_segment_from_midpoint(np.array([0, 0]), angle, 1.0)
            for angle in [0, 45, 90, 135]
        ]
    else:
        raise ValueError(f"Unknown principle: {principle}")

    # Step 1: Rotate each segment to align with nearest reference (modulo 90)
    for i in range(len(segments)):
        best_error = None
        best_rotation = None
        for ref in references:
            diff = geometry.angle_difference(
                geometry.orientation(ref), geometry.orientation(segments[i])
            )
            if best_error is None or abs(diff) < best_error:
                best_error = abs(diff)
                best_rotation = -diff
        segments[i] = geometry.rotate(segments[i], best_rotation)

    if visualize:
        plotting.segments(segments, title="After Angle Regularization")

    # Step 2: Merge parallel segments that are close together
    n = len(segments)
    i = 0
    while i < n:
        s = segments[i]
        t = segments[(i + 1) % n]

        if segment_offset(s, t) < max_offset and is_parallel(s, t):
            # Merge the two segments
            orient = geometry.orientation(s)
            minx = min(s[0], s[2], t[0], t[2])
            maxx = max(s[0], s[2], t[0], t[2])
            miny = min(s[1], s[3], t[1], t[3])
            maxy = max(s[1], s[3], t[1], t[3])
            seg_len = np.linalg.norm([maxx - minx, maxy - miny])
            mid = np.array([(minx + maxx) / 2.0, (miny + maxy) / 2.0])
            merged = make_segment_from_midpoint(mid, orient, seg_len)

            segments[i] = merged
            del segments[(i + 1) % n]
            n -= 1
        else:
            i += 1

    if visualize:
        plotting.segments(segments, title="After Merging Parallel Segments")

    # Step 3: Insert link segments between parallel consecutive segments
    n = len(segments)
    segs_with_links = []
    for i in range(n):
        seg_i = segments[i]
        seg_j = segments[(i + 1) % n]
        segs_with_links.append(seg_i)

        if is_parallel(seg_i, seg_j):
            # Create a link segment between the two parallel segments
            i_target = np.array([seg_i[2], seg_i[3]])
            j_source = np.array([seg_j[0], seg_j[1]])

            p = project_point_onto_line((j_source[0], j_source[1]), seg_i)
            q = project_point_onto_line((i_target[0], i_target[1]), seg_j)

            p = np.array(p)
            q = np.array(q)

            si = (p + i_target) / 2.0
            ti = (q + j_source) / 2.0

            link = np.array([si[0], si[1], ti[0], ti[1]], dtype=np.float64)
            segs_with_links.append(link)

    if visualize:
        plotting.segments(segs_with_links, title="With Link Segments")

    # Step 4: Compute intersection points to form the regularized contour
    n = len(segs_with_links)
    regularized_points = []
    for i in range(n):
        s = segs_with_links[i]
        t = segs_with_links[(i + 1) % n]
        intersection = line_intersection(s, t)
        if intersection is not None:
            regularized_points.append(intersection)
        else:
            # Fallback: use endpoint
            regularized_points.append((s[2], s[3]))

    if visualize:
        # Plot the final regularized contour
        if regularized_points:
            closed = regularized_points + [regularized_points[0]]
            xs, ys = zip(*closed)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(xs, ys, "k-", linewidth=1.5)
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.title("Regularized Contour")
            plt.show()

    return np.array(regularized_points)


def load_polylines(filename: str | None = None) -> List[Tuple[float, float]]:
    """Load points from a polylines file.

    Format: "2 x1 y1 z1 x2 y2 z2" per line (segments)
    We extract unique points from the segments.

    Args:
        filename: Path to polylines file. If None, loads bundled contour.polylines.
    """
    if filename is None:
        # Load bundled data file
        data_file = importlib.resources.files(__package__).joinpath("contour.polylines")
        content = data_file.read_text()
        lines = content.strip().split("\n")
    else:
        with open(filename) as f:
            lines = f.readlines()

    points = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            x1, y1 = float(parts[1]), float(parts[2])
            if not points or (points[-1][0] != x1 or points[-1][1] != y1):
                points.append((x1, y1))
    return points


