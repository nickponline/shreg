#!/usr/bin/env python3
"""Generate example plots for the README."""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from shreg import (
    solve_line_segments,
    regularize_contour,
    process_real,
    create_example_angles,
    create_cgal_example,
    seg,
    plotting,
)

# Disable interactive plotting
plotting.disable()


def main():
    print("Generating README plots...")

    # 1. Segment regularization - Real data
    print("  Generating segment_real_data.png...")
    segments = process_real()
    before = [s.copy() for s in segments]
    after = solve_line_segments(
        segments, offset=True, angle=True, maximum_angle=80, maximum_offset=2.0
    )
    plotting.comparison(before, after, title="Segment Regularization - Real Data",
                        show=False, save_path="docs/images/segment_real_data.png")

    # 2. Angle regularization
    print("  Generating segment_angle.png...")
    segments = create_example_angles()
    before = [s.copy() for s in segments]
    after = solve_line_segments(segments, offset=False, angle=True, maximum_angle=40)
    plotting.comparison(before, after, title="Angle Regularization",
                        show=False, save_path="docs/images/segment_angle.png")

    # 3. Hexagon example
    print("  Generating segment_hexagon.png...")
    segments = [
        seg(0.2, 0.0, 1.2, 0.0),
        seg(1.2, 0.1, 2.2, 0.1),
        seg(2.2, 0.0, 2.0, 2.0),
        seg(2.0, 2.0, 1.0, 2.0),
        seg(1.0, 1.9, 0.0, 1.9),
        seg(0.0, 2.0, 0.2, 0.0),
    ]
    before = [s.copy() for s in segments]
    after = solve_line_segments(
        segments, offset=True, angle=True, maximum_offset=0.5, maximum_angle=25
    )
    plotting.comparison(before, after, title="Hexagon - Combined Regularization",
                        show=False, save_path="docs/images/segment_hexagon.png")

    # 4. CGAL 2.4 Angle + Offset Regularization example
    print("  Generating segment_cgal_example.png...")
    segments, _groups = create_cgal_example()
    before = [s.copy() for s in segments]
    after = solve_line_segments(
        segments, offset=True, angle=True, maximum_offset=0.1, maximum_angle=10
    )
    plotting.comparison(before, after, title="CGAL 2.4 - Angle + Offset Regularization",
                        show=False, save_path="docs/images/segment_cgal_example.png")

    # 5. Contour regularization - Simple polygon
    print("  Generating contour_simple.png...")
    points = [
        [45, 29], [65, 440], [44, 498], [446, 498], [429, 325],
        [499, 309], [448, 206], [479, 148], [479, 31], [247, 88],
    ]
    result = regularize_contour(points, "axis", max_offset=20, visualize=False)
    result_list = [(float(p[0]), float(p[1])) for p in result]
    plotting.contour_comparison(
        [(float(p[0]), float(p[1])) for p in points],
        result_list,
        title="Contour Regularization - Axis Aligned",
        show=False,
        save_path="docs/images/contour_simple.png"
    )

    # 6. Contour regularization - Rectangle-like shape
    print("  Generating contour_rectangle.png...")
    points = [
        [101, 372], [101, 344], [97, 280], [114, 229], [112, 192],
        [108, 132], [104, 85], [112, 61], [191, 49], [247, 43],
        [281, 47], [310, 59], [340, 56], [406, 56], [427, 53],
        [436, 76], [437, 112], [434, 142], [434, 172], [435, 188],
        [432, 225], [432, 227], [432, 246], [431, 276], [435, 315],
        [433, 332], [429, 389], [394, 397], [360, 385], [335, 384],
        [336, 374], [336, 351], [336, 299], [338, 246], [349, 208],
        [338, 156], [339, 129], [288, 119], [273, 119], [232, 118],
        [210, 121], [172, 125], [187, 154], [180, 164], [180, 183],
        [180, 194], [178, 225], [178, 248], [178, 284], [176, 312],
        [172, 362], [172, 373],
    ]
    result = regularize_contour(points, "axis", max_offset=20, visualize=False)
    result_list = [(float(p[0]), float(p[1])) for p in result]
    plotting.contour_comparison(
        [(float(p[0]), float(p[1])) for p in points],
        result_list,
        title="Contour Regularization - Complex Shape",
        show=False,
        save_path="docs/images/contour_rectangle.png"
    )

    print("Done! Plots saved to docs/images/")


if __name__ == "__main__":
    main()
