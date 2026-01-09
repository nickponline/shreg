# API Reference

## Segment Regularization

### `solve_line_segments(segments, offset=True, angle=True, maximum_offset=0.5, maximum_angle=25)`

Regularize a list of line segments.

**Parameters:**
- `segments`: List of segments, where each segment is a numpy array `[x1, y1, x2, y2]`
- `offset`: Whether to regularize offsets (default: `True`)
- `angle`: Whether to regularize angles (default: `True`)
- `maximum_offset`: Maximum offset tolerance in units (default: `0.5`)
- `maximum_angle`: Maximum angle tolerance in degrees (default: `25`)

**Returns:** List of regularized segments

### `seg(x1, y1, x2, y2)`

Helper function to create a segment array.

```python
from shreg import seg
s = seg(0, 0, 1, 1)  # Creates np.array([0, 0, 1, 1])
```

## Snap Regularization

### `snap_regularize_segments(segments, epsilon=1.0, method="hard", soft_weight=100.0, t_junctions=False)`

Connect nearby endpoints by snapping them together.

**Parameters:**
- `segments`: List of segments, where each segment is a numpy array `[x1, y1, x2, y2]`
- `epsilon`: Maximum distance for endpoints to be considered "close" and snapped (default: `1.0`)
- `method`: Snapping method to use:
  - `"cluster"`: Fast centroid-based method (recommended for most cases)
  - `"hard"`: QP with exact equality constraints (perfectly watertight)
  - `"soft"`: QP with spring penalty (elastic connections)
- `soft_weight`: Spring stiffness for soft constraints. Higher values = stiffer springs (default: `100.0`)
- `t_junctions`: Whether to detect and snap T-junctions (default: `False`)

**Returns:** List of segments with snapped endpoints

### `find_endpoint_clusters(segments, epsilon=1.0)`

Find clusters of nearby endpoints using spatial indexing.

**Parameters:**
- `segments`: List of segments
- `epsilon`: Maximum distance for clustering

**Returns:** List of clusters, where each cluster is a list of `(segment_idx, endpoint_idx)` tuples

### `find_t_junctions(segments, epsilon=1.0, exclude_clusters=None)`

Find T-junctions where endpoints are close to segment interiors.

**Parameters:**
- `segments`: List of segments
- `epsilon`: Maximum distance for T-junction detection
- `exclude_clusters`: Endpoint clusters to exclude (already handled by endpoint-to-endpoint snapping)

**Returns:** List of `((segment_idx, endpoint_idx), target_segment_idx)` tuples

## Metric Regularization

### `metric_regularize_segments(segments, equal_length=True, length_quantization=False, equal_spacing=True, base_unit=1.0, length_tolerance=0.1, quantization_tolerance=0.3, angle_tolerance=5.0, max_iterations=3)`

Regularize segments using metric and pattern constraints.

**Parameters:**
- `segments`: List of segments, where each segment is a numpy array `[x1, y1, x2, y2]`
- `equal_length`: Force segments with similar lengths to be exactly equal (default: `True`)
- `length_quantization`: Snap lengths to multiples of `base_unit` (default: `False`)
- `equal_spacing`: Force equal gaps between parallel lines (default: `True`)
- `base_unit`: Base unit for length quantization, e.g., 1.0 meter (default: `1.0`)
- `length_tolerance`: Relative tolerance for equal length detection (default: `0.1` = 10%)
- `quantization_tolerance`: Tolerance for quantization as fraction of `base_unit` (default: `0.3`)
- `angle_tolerance`: Maximum angle difference in degrees to consider lines parallel (default: `5.0`)
- `max_iterations`: Maximum SQP iterations for iterative refinement (default: `3`)

**Returns:** List of regularized segments

**Note:** Uses linearization since length calculation is non-linear. The algorithm iteratively refines the solution using Sequential Quadratic Programming (SQP).

### `find_equal_length_pairs(segments, tolerance=0.1, min_length=0.5)`

Find pairs of segments with similar lengths.

**Parameters:**
- `segments`: List of segments
- `tolerance`: Maximum relative length difference to consider "similar" (default: `0.1`)
- `min_length`: Minimum segment length to consider (default: `0.5`)

**Returns:** List of `(segment_idx_a, segment_idx_b)` tuples

### `find_length_quantization_targets(segments, base_unit=1.0, tolerance=0.3, min_length=0.5)`

Find segments whose lengths should be quantized to multiples of `base_unit`.

**Parameters:**
- `segments`: List of segments
- `base_unit`: Base unit for quantization (default: `1.0`)
- `tolerance`: Maximum distance from nearest multiple as fraction of `base_unit` (default: `0.3`)
- `min_length`: Minimum segment length to consider (default: `0.5`)

**Returns:** List of `(segment_idx, target_length)` tuples

### `find_parallel_line_groups(segments, angle_tolerance=5.0, min_group_size=3)`

Find groups of parallel segments for equal spacing regularization.

**Parameters:**
- `segments`: List of segments
- `angle_tolerance`: Maximum angle difference in degrees to consider parallel (default: `5.0`)
- `min_group_size`: Minimum number of segments to form a group (default: `3`)

**Returns:** List of groups, where each group is a list of segment indices

## Contour Regularization

### `regularize_contour(points, principle="longest", max_offset=20.0, visualize=False)`

Regularize a closed contour.

**Parameters:**
- `points`: List of `[x, y]` coordinates forming a closed polygon
- `principle`: How to determine principal directions:
  - `"longest"`: Use the longest edge as reference
  - `"axis"`: Align to horizontal/vertical axes
  - `"cardinal"`: Use 0, 45, 90, 135 degree directions
- `max_offset`: Maximum offset for merging parallel segments (default: `20.0`)
- `visualize`: Whether to show intermediate plots (default: `False`)

**Returns:** numpy array of regularized points

### `load_polylines(filename=None)`

Load contour points from a polylines file.

**Parameters:**
- `filename`: Path to polylines file. If `None`, loads bundled example data.

## Geometry Utilities

The package includes various geometry functions:

```python
from shreg import (
    length,              # Segment length
    midpoint,            # Segment midpoint
    orientation,         # Segment orientation in degrees [0, 180)
    direction,           # Unit direction vector
    normal,              # Unit normal vector
    rotate,              # Rotate segment around midpoint
    translate_by_normal, # Translate along normal
    angle_difference,    # Angle difference modulo 90 degrees
)
```

## Plotting Utilities

```python
from shreg import plotting

# Enable/disable all plotting
plotting.enable()
plotting.disable()

# Before/after comparison
plotting.comparison(before_segments, after_segments, title="My Title")

# Context manager for before/after
with plotting.BeforeAfter("My Title") as plot:
    plot.before(segments)
    segments = regularize(segments)
    plot.after(segments)

# Save plots to file
plotting.comparison(before, after, save_path="output.png")
```
