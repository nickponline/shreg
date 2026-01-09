# shreg - Shape Regularization

A Python implementation of various shape regularization algorithms for regularizing line segments and closed contours.

Shape regularization is a technique used in computational geometry to clean up noisy or imprecise geometric data by aligning segments to common orientations and adjusting their positions to create cleaner, more regular shapes.

## Features

- **Segment Regularization**: Align line segments to common angles and offsets using quadratic programming optimization
- **Snap Regularization**: Connect nearby endpoints to create watertight polygons and meshes
- **Contour Regularization**: Simplify closed polygons by aligning edges to principal directions
- **T-Junction Detection**: Snap endpoints onto segment interiors for proper connectivity
- **Flexible Configuration**: Control maximum angle and offset tolerances
- **Visualization**: Built-in plotting utilities for before/after comparisons
- **Pure Python**: No dependencies required

## Installation

### Using pip

```bash
pip install shreg
```

### Using uv

```bash
uv pip install shreg
```

### From source

```bash
git clone https://github.com/nickp/shreg.git
cd shreg
pip install -e .
```

## Quick Start

### Segment Regularization

Regularize a set of line segments by aligning their angles and offsets:

```python
import numpy as np
from shreg import solve_line_segments, seg

# Create some segments (each segment is [x1, y1, x2, y2])
segments = [
    seg(0.0, 0.0, 1.0, 0.02),   # Nearly horizontal
    seg(0.0, 1.0, 1.0, 1.05),   # Nearly horizontal, slightly offset
    seg(1.0, 0.0, 1.02, 1.0),   # Nearly vertical
]

# Regularize: align angles within 25 degrees, offsets within 0.5 units
result = solve_line_segments(
    segments,
    angle=True,
    offset=True,
    maximum_angle=25,
    maximum_offset=0.5
)
```

### Snap Regularization

Close gaps between nearby endpoints to create watertight polygons:

```python
from shreg import snap_regularize_segments, seg

# Create segments with small gaps at corners
segments = [
    seg(0.0, 0.0, 1.0, 0.0),    # bottom edge
    seg(1.05, 0.02, 1.0, 1.0),  # right edge (gap at bottom-right)
    seg(1.0, 1.03, 0.0, 0.98),  # top edge (gap at corners)
    seg(-0.02, 1.0, 0.0, 0.0),  # left edge (gap at top-left)
]

# Snap endpoints within 0.1 units of each other
result = snap_regularize_segments(
    segments,
    epsilon=0.1,      # Distance threshold for snapping
    method="cluster"  # Fast centroid-based method
)
# Result: All corners are now perfectly connected
```

### Contour Regularization

Simplify a closed polygon by aligning edges to principal directions:

```python
from shreg import regularize_contour

# Define a noisy polygon (list of [x, y] points)
points = [
    [45, 29], [65, 440], [44, 498], [446, 498], [429, 325],
    [499, 309], [448, 206], [479, 148], [479, 31], [247, 88],
]

# Regularize with axis alignment
result = regularize_contour(
    points,
    principle="axis",     # Align to horizontal/vertical
    max_offset=20,        # Maximum offset for merging
)

print(f"Simplified from {len(points)} to {len(result)} points")
```

## Examples

### Segment Regularization

The algorithm optimizes segment orientations and positions to create cleaner line arrangements:

![Segment Regularization - Real Data](docs/images/segment_real_data.png)

Angle regularization aligns crossing lines to common orientations:

![Angle Regularization](docs/images/segment_angle.png)

Combined angle and offset regularization on a hexagon:

![Segment Regularization](docs/images/segment_hexagon.png)

#### Angle + Offset Regularization with Groups

This example from the [CGAL documentation](https://doc.cgal.org/latest/Shape_regularization/index.html#title10) demonstrates sequential angle and offset regularization on 15 segments organized into three groups: outer boundary, top rhombus, and bottom rhombus.

```python
from shreg import solve_line_segments, create_cgal_example

# Load the 15 segments from the CGAL example
segments, groups = create_cgal_example()

# Regularize with tight tolerances
result = solve_line_segments(
    segments,
    angle=True,
    offset=True,
    maximum_angle=10,    # 10 degrees max angle deviation
    maximum_offset=0.1   # 0.1 units max offset
)
```

![CGAL 2.4 Angle + Offset Regularization](docs/images/segment_cgal_example.png)

### Snap Regularization

Snap regularization connects nearby endpoints to create watertight geometry. This is essential for creating closed polygons suitable for 3D extrusion, mesh generation, or CAD operations.

#### Cluster Method (Fastest)

The cluster method groups nearby endpoints and moves them to their centroid. This is the fastest approach and guarantees watertight results:

```python
from shreg import snap_regularize_segments, seg

segments = [
    seg(0.0, 0.0, 1.0, 0.05),
    seg(1.08, 0.0, 1.05, 1.0),
    seg(1.0, 1.08, 0.0, 0.95),
    seg(-0.05, 1.0, 0.0, 0.0),
]
result = snap_regularize_segments(segments, epsilon=0.15, method="cluster")
```

![Snap Regularization - Cluster Method](docs/images/snap_cluster.png)

#### Hard Constraints (Exact)

Hard constraints use quadratic programming to find the optimal positions that exactly satisfy all snap constraints while minimizing total endpoint movement:

```python
result = snap_regularize_segments(segments, epsilon=0.15, method="hard")
```

![Snap Regularization - Hard Constraints](docs/images/snap_hard.png)

#### Soft Constraints (Elastic)

Soft constraints add "spring" forces between endpoints that should connect. This is useful when data is noisy and you're not certain endpoints should be exactly coincident:

```python
result = snap_regularize_segments(
    segments,
    epsilon=0.25,
    method="soft",
    soft_weight=50.0  # Higher = stiffer springs
)
```

![Snap Regularization - Soft Constraints](docs/images/snap_soft.png)

#### T-Junction Detection

T-junctions occur when an endpoint should snap onto another segment's interior (not its endpoints). Enable T-junction detection for proper connectivity:

```python
segments = [
    seg(0.0, 0.0, 2.0, 0.0),      # horizontal line
    seg(0.0, 1.0, 2.0, 1.0),      # horizontal line
    seg(0.95, -0.08, 1.05, 1.1),  # vertical line (forms T-junctions)
]
result = snap_regularize_segments(
    segments, epsilon=0.15, method="cluster", t_junctions=True
)
```

![Snap Regularization - T-Junctions](docs/images/snap_tjunction.png)

#### Complex Polygons

Snap regularization works on polygons of any complexity:

![Snap Regularization - Complex Polygon](docs/images/snap_complex.png)

### Contour Regularization

Simplify complex polygons while preserving their essential shape:

![Contour Regularization - Simple](docs/images/contour_simple.png)

Complex shapes are reduced to their essential vertices:

![Contour Regularization - Complex](docs/images/contour_rectangle.png)

## API Reference

### Segment Regularization

#### `solve_line_segments(segments, offset=True, angle=True, maximum_offset=0.5, maximum_angle=25)`

Regularize a list of line segments.

**Parameters:**
- `segments`: List of segments, where each segment is a numpy array `[x1, y1, x2, y2]`
- `offset`: Whether to regularize offsets (default: `True`)
- `angle`: Whether to regularize angles (default: `True`)
- `maximum_offset`: Maximum offset tolerance in units (default: `0.5`)
- `maximum_angle`: Maximum angle tolerance in degrees (default: `25`)

**Returns:** List of regularized segments

#### `seg(x1, y1, x2, y2)`

Helper function to create a segment array.

```python
from shreg import seg
s = seg(0, 0, 1, 1)  # Creates np.array([0, 0, 1, 1])
```

### Snap Regularization

#### `snap_regularize_segments(segments, epsilon=1.0, method="hard", soft_weight=100.0, t_junctions=False)`

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

#### `find_endpoint_clusters(segments, epsilon=1.0)`

Find clusters of nearby endpoints using spatial indexing.

**Parameters:**
- `segments`: List of segments
- `epsilon`: Maximum distance for clustering

**Returns:** List of clusters, where each cluster is a list of `(segment_idx, endpoint_idx)` tuples

#### `find_t_junctions(segments, epsilon=1.0, exclude_clusters=None)`

Find T-junctions where endpoints are close to segment interiors.

**Parameters:**
- `segments`: List of segments
- `epsilon`: Maximum distance for T-junction detection
- `exclude_clusters`: Endpoint clusters to exclude (already handled by endpoint-to-endpoint snapping)

**Returns:** List of `((segment_idx, endpoint_idx), target_segment_idx)` tuples

### Contour Regularization

#### `regularize_contour(points, principle="longest", max_offset=20.0, visualize=False)`

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

#### `load_polylines(filename=None)`

Load contour points from a polylines file.

**Parameters:**
- `filename`: Path to polylines file. If `None`, loads bundled example data.

### Geometry Utilities

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

### Plotting Utilities

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

## Command Line Interface

Run the demo examples:

```bash
# Run all examples with visualization
shreg

# Run without visualization (batch mode)
shreg --no-plot

# Run only segment examples
shreg --segments

# Run only contour examples
shreg --contours
```

Or using Python module syntax:

```bash
python -m shreg --help
```

## Algorithm

### Energy Minimization via Quadratic Programming

The regularization problem is formulated as an **energy minimization** problem. Given a set of segments, we seek small adjustments (rotations and translations) that minimize an energy function while respecting constraints on maximum deviations.

The energy function balances two objectives:
- **Fidelity**: Keep segments close to their original positions
- **Regularity**: Encourage nearby segments to share common angles and offsets

This leads to a **quadratic program (QP)** of the form:

```
minimize    (1/2) x'Px + q'x
subject to  l <= Ax <= u
```

where `x` contains the rotation and translation corrections for each segment, `P` encodes the fidelity cost, and the constraints enforce that angle/offset differences between nearby segments are minimized.

### Segment Regularization Pipeline

1. **Neighbor Detection**: Use Delaunay triangulation on segment midpoints to identify nearby segment pairs efficiently
2. **Constraint Graph**: Build constraints for angle and offset differences between neighboring segments within tolerance bounds
3. **QP Optimization**: Solve the quadratic program using OSQP to find optimal corrections
4. **Application**: Apply computed rotations and translations to each segment

### Snap (Connectivity) Regularization

Snap regularization is formulated as a **Constrained Quadratic Programming** problem that minimizes endpoint movement while enforcing connectivity constraints.

**Variables:** For N segments, the state vector contains all 4N endpoint coordinates:
```
x = [x₁₁, y₁₁, x₁₂, y₁₂, ..., xₙ₂, yₙ₂]ᵀ
```

**Objective (Fidelity):** Minimize squared distance from original positions:
```
minimize (1/2) Σᵢ (||uᵢ - ûᵢ||² + ||vᵢ - v̂ᵢ||²)
```

**Methods:**

| Method | Formulation | Use Case |
|--------|-------------|----------|
| `cluster` | Replace clustered endpoints with centroid | Fast, guaranteed watertight |
| `hard` | Equality constraints: vᵢ - uⱼ = 0 | Exact connections required |
| `soft` | Penalty term: λ·Σ||vᵢ - uⱼ||² | Noisy data, uncertain connections |

**Pipeline:**
1. **Endpoint Detection**: Build KD-Tree on all 2N endpoints
2. **Clustering**: Use Union-Find to group endpoints within ε distance
3. **Variable Reduction** (cluster): Replace clusters with single variables
4. **QP Solve** (hard/soft): Optimize using OSQP
5. **T-Junction Handling**: Project endpoints onto target segments

### Contour Regularization

The contour regularization algorithm follows CGAL's approach for closed polygons:

1. **Angle Alignment**: Rotate each edge to align with principal directions (modulo 90 degrees)
2. **Parallel Merging**: Merge consecutive parallel edges that are close together
3. **Link Insertion**: Insert connecting segments between remaining parallel edges
4. **Intersection**: Compute intersection points to form the final regularized polygon

## Dependencies

- `numpy >= 1.20.0`
- `scipy >= 1.7.0`
- `osqp >= 0.6.0`
- `matplotlib >= 3.5.0`

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/ -v
```

## References

- Jean-Philippe Bauchet and Florent Lafarge. **KIPPI: KInetic Polygonal Partitioning of Images**. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 3146–3154, Salt Lake City, United States, June 2018. [[PDF](https://hal.inria.fr/hal-01741686/document)]
- [CGAL Shape Regularization Documentation](https://doc.cgal.org/latest/Shape_regularization/)
- [OSQP: Operator Splitting Quadratic Program Solver](https://osqp.org/)
