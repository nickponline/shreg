"""Shape Regularization - A Python implementation of CGAL Shape Regularization.

This package provides algorithms for regularizing line segments and contours,
based on the CGAL Shape Regularization library.

Reference: https://doc.cgal.org/latest/Shape_regularization/
"""
from __future__ import annotations

__version__ = "0.1.0"

# Core segment regularization
from .segments_reg import (
    solve_line_segments,
    create_cgal_example,
    create_example_angles,
    create_example_offsets,
    process_real,
    seg,
)

# Contour regularization
from .contours_reg import (
    regularize_contour,
    load_polylines,
)

# Geometry utilities
from .geometry import (
    Segment,
    Point,
    length,
    midpoint,
    orientation,
    direction,
    normal,
    rotate,
    translate_by_normal,
    flip,
    standardize_segment,
    angle_difference,
    signed_distance_to_point,
    validate_segment,
)

# Plotting utilities
from . import plotting

__all__ = [
    # Version
    "__version__",
    # Segment regularization
    "solve_line_segments",
    "create_cgal_example",
    "create_example_angles",
    "create_example_offsets",
    "process_real",
    "seg",
    # Contour regularization
    "regularize_contour",
    "load_polylines",
    # Geometry types
    "Segment",
    "Point",
    # Geometry functions
    "length",
    "midpoint",
    "orientation",
    "direction",
    "normal",
    "rotate",
    "translate_by_normal",
    "flip",
    "standardize_segment",
    "angle_difference",
    "signed_distance_to_point",
    "validate_segment",
    # Plotting
    "plotting",
]
