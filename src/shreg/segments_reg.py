"""Segment regularization using quadratic programming.

This module implements the CGAL Shape Regularization algorithm for line segments,
which regularizes angles and offsets between nearby segments using OSQP optimization.

Reference: https://doc.cgal.org/latest/Shape_regularization/
"""
from __future__ import annotations

import importlib.resources
import random
from typing import Callable, List

import numpy as np
import osqp
from scipy import sparse
from scipy.spatial import Delaunay

from . import geometry
from . import logger
from . import plotting
from .geometry import Segment

log = logger.get_custom_logger(__name__)


def generate_random_point_on_line_segment(
    segment: Segment, i: int, n: int
) -> tuple[float, float]:
    x1, y1, x2, y2 = segment
    t = (i + 1) / n
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return x, y


def create_example_angles() -> List[Segment]:
    segments: List[Segment] = []

    s1 = np.array([-250, -50, -250, 50], dtype=np.float64)
    s2 = np.array([250, -250, 250, 250], dtype=np.float64)
    points1 = [generate_random_point_on_line_segment(s1, i, 50) for i in range(50)]
    points2 = [generate_random_point_on_line_segment(s2, i, 50) for i in range(50)]
    points1 = sorted(points1, key=lambda x: x[1])
    points2 = sorted(points2, key=lambda x: x[1])
    for i in range(50):
        segments.append(
            np.array([
                points1[i][0] + random.random() / 1000,
                points1[i][1],
                points2[i][0],
                points2[i][1],
            ], dtype=np.float64)
        )

    s1 = np.array([-50, -250, 50, -250], dtype=np.float64)
    s2 = np.array([-250, 250, 250, 250], dtype=np.float64)
    points1 = [generate_random_point_on_line_segment(s1, i, 50) for i in range(50)]
    points2 = [generate_random_point_on_line_segment(s2, i, 50) for i in range(50)]
    points1 = sorted(points1, key=lambda x: x[0])
    points2 = sorted(points2, key=lambda x: x[0])
    for i in range(50):
        segments.append(
            np.array([
                points1[i][0],
                points1[i][1] + random.random() / 1000,
                points2[i][0],
                points2[i][1],
            ], dtype=np.float64)
        )
    return segments


def get_coefficient_value(theta: float, iterator: float) -> float:
    if (
        theta == 0.0
        or theta == np.pi / 2.0
        or theta == np.pi
        or theta == 3.0 * np.pi / 2.0
    ):
        iterator = 0.0
    elif (
        theta == np.pi / 4.0
        or theta == 3.0 * np.pi / 4.0
        or theta == 5.0 * np.pi / 4.0
        or theta == 7.0 * np.pi / 4.0
    ):
        iterator = 0.22
    elif (
        (theta > 0.0 and theta < np.pi / 4.0)
        or (theta > np.pi / 2.0 and theta < 3.0 * np.pi / 4.0)
        or (theta > np.pi and theta < 5.0 * np.pi / 4.0)
        or (theta > 3.0 * np.pi / 2.0 and theta < 7.0 * np.pi / 4.0)
    ):
        iterator += 0.02
    else:
        iterator -= 0.02

    if theta < np.pi:
        return -1.0 * iterator
    else:
        return iterator


def seg(x1: float, y1: float, x2: float, y2: float) -> Segment:
    """Helper function to create a segment array."""
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def create_cgal_example() -> tuple[List[Segment], List[List[int]]]:
    """Create the CGAL 2.4 Angle + Offset Regularization example.

    This example from the CGAL documentation shows 15 segments organized into
    three contextual groups:
    - Outer boundary (7 segments): indices 0-6
    - Top rhombus (4 segments): indices 7-10
    - Bottom rhombus (4 segments): indices 11-14

    Reference: https://doc.cgal.org/latest/Shape_regularization/index.html#title10

    Returns:
        Tuple of (segments, groups) where groups define which segments
        should be regularized together.
    """
    segments = [
        # Outer boundary
        seg(1.000000, 1.000000, 0.925377, 2.995179),
        seg(1.000000, 3.000000, 1.066662, 4.951894),
        seg(1.000000, 5.000000, 2.950000, 4.930389),
        seg(3.000000, 4.950000, 2.934996, 3.008203),
        seg(3.085452, 3.003266, 2.969782, 1.002004),
        seg(0.948866, 3.033161, 2.900000, 3.000000),
        seg(0.930000, 1.000000, 2.860000, 1.002004),
        # Top rhombus
        seg(1.600000, 4.000000, 1.932136, 4.364718),
        seg(1.598613, 3.982686, 2.018220, 3.686595),
        seg(1.951872, 4.363094, 2.290848, 4.054154),
        seg(2.018220, 3.686595, 2.304517, 4.045054),
        # Bottom rhombus
        seg(1.642059, 1.928505, 1.993860, 2.247986),
        seg(1.993860, 2.247986, 2.259099, 1.919966),
        seg(1.629845, 1.923077, 1.968759, 1.599174),
        seg(2.259099, 1.919966, 1.968759, 1.599170),
    ]
    groups = [
        [0, 1, 2, 3, 4, 5, 6],  # Outer boundary
        [7, 8, 9, 10],          # Top rhombus
        [11, 12, 13, 14],       # Bottom rhombus
    ]
    return segments, groups


def create_example_offsets() -> List[Segment]:
    segments: List[Segment] = []

    theta = 0.0
    iterator = 0.0
    theta_step = np.pi / 25.0

    while theta < 2.0 * np.pi:
        st = np.sin(theta)
        ct = np.cos(theta)

        a = [0.0, 0.0]
        b = [ct, st]

        coef = get_coefficient_value(theta, iterator)
        c = [ct, st + coef]
        d = [2.0 * ct, 2.0 * st + coef]
        theta += theta_step

        segments.append(np.array([a[0], a[1], b[0], b[1]], dtype=np.float64))
        segments.append(np.array([c[0], c[1], d[0], d[1]], dtype=np.float64))
    return segments


def get_nearest_neighbour(points: np.ndarray) -> List[tuple[int, int]]:
    """Calculate nearest neighbour for each point."""
    n = len(points)
    nearest_neighbour = np.zeros(n, dtype=int)
    nearest_distance = np.zeros(n, dtype=float)
    for i in range(n):
        nearest_distance[i] = np.inf
        for j in range(n):
            if i == j:
                continue
            d = np.linalg.norm(points[i] - points[j])
            if d < nearest_distance[i]:
                nearest_distance[i] = d
                nearest_neighbour[i] = j

    ret = [(i, nearest_neighbour[i]) for i in range(n)]
    ret = set([tuple(sorted(x)) for x in ret])
    return list(ret)


def angle_error(s1: Segment, s2: Segment) -> float:
    return geometry.angle_difference(geometry.orientation(s1), geometry.orientation(s2))


def get_reference(frame: List[float], line: Segment) -> List[float]:
    mid = geometry.midpoint(line)
    dx = mid[0] - frame[0]
    dy = mid[1] - frame[1]
    angle_rad = np.radians(geometry.orientation(line))
    sin_val = np.sin(angle_rad)
    cos_val = np.cos(angle_rad)

    rx = dx * cos_val + dy * sin_val
    ry = dy * cos_val - dx * sin_val
    return [rx, ry]


def offset_error(s1: Segment, s2: Segment, debug: bool = False) -> float:
    frame = [0, 0]
    ref1 = get_reference(frame, s1)
    ref2 = get_reference(frame, s2)

    diff_orientation = abs(geometry.orientation(s1) - geometry.orientation(s2))

    if diff_orientation < 0.5:
        return ref2[1] - ref1[1]
    else:
        return np.inf


def solve(
    segments: List[Segment],
    edges: List[tuple[int, int]],
    max_bound: float = 25,
    error_func: Callable[[Segment, Segment], float] | None = None,
    update_func: Callable[[List[Segment], int, float], None] | None = None,
    debug: bool = False,
) -> List[Segment]:
    weight = 100000
    lam = 4 / 5.0
    neg_inf = -1e12
    pos_inf = 1e12
    val_neg = -2 * lam
    val_pos = +2 * lam

    k = len(segments)
    e = len(edges)
    n = k + e
    m = 2 * e + n

    if debug:
        log.info(f"{k} segments, {e} edges")
    if debug:
        log.info(f"{n} variables, {m} constraints")

    D = np.zeros((n, n))
    q = np.zeros(n)

    for i in range(n):
        if i < k:
            D[i, i] = 2.0 * weight * (1 - lam) / (max_bound * max_bound * k)
        else:
            D[i, i] = 0.0

    for i in range(n):
        if i >= k:
            q[i] = lam * weight / (4 * max_bound * e)
        else:
            q[i] = 0.0

    P = sparse.csc_matrix(D)

    X = np.zeros((m, n))
    ij = k
    it = 0

    for i, j in edges:
        X[it, i] = val_neg
        if debug:
            log.info(f"{it} {i} {val_neg}")
        X[it, j] = val_pos
        if debug:
            log.info(f"{it} {j} {val_pos}")
        X[it, ij] = -1
        if debug:
            log.info(f"{it} {ij} -1")
        it += 1

        X[it, i] = val_pos
        if debug:
            log.info(f"{it} {i} {val_pos}")
        X[it, j] = val_neg
        if debug:
            log.info(f"{it} {j} {val_neg}")
        X[it, ij] = -1
        if debug:
            log.info(f"{it} {ij} -1")
        ij += 1
        it += 1

    s = len(edges) * 2
    for i in range(n):
        X[s + i, i] = 1
        if debug:
            log.info(f"{s + i} {i} 1")

    A = sparse.csc_matrix(X)
    L = np.zeros(m, dtype="float64")
    U = np.zeros(m, dtype="float64")

    if debug:
        log.info(f"{L.shape} {U.shape}")

    tit = 0
    for i in range(m):
        if i < 2 * e:
            edge = edges[tit]
            value = error_func(segments[edge[0]], segments[edge[1]])
            if i % 2 == 0:
                L[i] = neg_inf
                U[i] = val_neg * value
            else:
                L[i] = neg_inf
                U[i] = val_pos * value
                tit += 1
        elif i < 2 * e + k:
            L[i] = -(1) * max_bound
            U[i] = +(1) * max_bound
        else:
            L[i] = neg_inf
            U[i] = pos_inf

    prob = osqp.OSQP()

    try:
        prob.setup(P, q, A, L, U, eps_abs=1e-05, eps_rel=1e-05, verbose=False)
    except Exception as ex:
        log.info(f"Problem is not feasible, check constraints: {ex}")
        return segments
    res = prob.solve()
    solution = res.x

    if debug:
        log.info(f"solution: {solution}")
    if debug:
        log.info(f"objective: {solution.T.dot(D).dot(solution) + q.dot(solution)}")

    for i in range(k):
        correction = solution[i]
        update_func(segments, i, correction)

    return segments


def angle_update_func(segments: List[Segment], idx: int, correction: float) -> None:
    segments[idx] = geometry.rotate(segments[idx], correction)


def offset_update_func(segments: List[Segment], idx: int, correction: float) -> None:
    segments[idx] = geometry.translate_by_normal(segments[idx], correction)


def get_edges(
    segments: List[Segment],
    error_func: Callable[[Segment, Segment], float],
    bound: float = 25,
    debug: bool = False,
) -> List[tuple[int, int]]:
    points = np.array([geometry.midpoint(s) for s in segments])
    try:
        tri = Delaunay(points)
    except Exception:
        edges = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                error = error_func(segments[i], segments[j])
                if abs(error) < 2 * bound:
                    edges.append((i, j))

        return edges

    tripoints = tri.simplices

    if debug:
        plotting.delaunay_debug(segments, tripoints, error_func)

    edges = set()
    for v1, v2, v3 in tripoints:
        a, b = min(v1, v2), max(v1, v2)
        target = error_func(segments[a], segments[b])
        if abs(target) < bound * 2:
            edges.add((a, b))
        a, b = min(v2, v3), max(v2, v3)
        target = error_func(segments[a], segments[b])
        if abs(target) < bound * 2:
            edges.add((a, b))
        a, b = min(v3, v1), max(v3, v1)
        target = error_func(segments[a], segments[b])
        if abs(target) < bound * 2:
            edges.add((a, b))

    return list(edges)


def solve_line_segments(
    segments: List[Segment],
    offset: bool = True,
    angle: bool = True,
    maximum_offset: float = 0.5,
    maximum_angle: float = 25,
    debug: bool = False,
) -> List[Segment]:
    if len(segments) < 2:
        return segments

    # Convert to numpy arrays if needed
    segments = [
        np.array(s, dtype=np.float64) if not isinstance(s, np.ndarray) else s.astype(np.float64)
        for s in segments
    ]

    # Standardize segments (ensure x1 <= x2)
    segments = [geometry.standardize_segment(s) for s in segments]

    if angle:
        edges = get_edges(segments, angle_error, bound=maximum_angle, debug=debug)
        if debug:
            plotting.constraints(segments, edges, angle_error, title="Angle Constraints")
        solve(
            segments,
            edges,
            max_bound=maximum_angle,
            error_func=angle_error,
            update_func=angle_update_func,
        )

    if offset:
        edges = get_edges(segments, offset_error, bound=maximum_offset, debug=debug)
        if debug:
            plotting.constraints(segments, edges, offset_error, title="Offset Constraints")
        segments = solve(
            segments,
            edges,
            max_bound=maximum_offset,
            error_func=offset_error,
            update_func=offset_update_func,
        )

    return segments


def fit_line_segment(points: np.ndarray) -> tuple[float, float, float, float]:
    """Fit a line segment to set of 2d points and return x1, y1, x2, y2."""
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    x1 = x.min()
    x2 = x.max()
    y1 = m * x1 + c
    y2 = m * x2 + c
    return x1, y1, x2, y2


def process_real(plot: bool = False) -> List[Segment]:
    """Load and process real segment data from bundled data file."""
    last_group = None
    segments: List[Segment] = []
    all_points: List[tuple[float, float]] = []
    points = []

    # Load data file from package resources
    data_file = importlib.resources.files(__package__).joinpath("real_data_2.xyzi")
    content = data_file.read_text()

    for line in content.strip().split("\n"):
        line = line.strip()
        parts = line.split()
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        i = int(parts[3])

        if plot:
            all_points.append((x, y))

        if i != last_group:
            if len(points) > 1:
                pts = np.array(points)
                x1, y1, x2, y2 = fit_line_segment(pts)
                segments.append(np.array([x1, y1, x2, y2], dtype=np.float64))

            last_group = i
            points = [(x, y)]
        else:
            points.append((x, y))

    if plot:
        plotting.points_and_segments(segments, all_points, title="Real Data")

    return segments


