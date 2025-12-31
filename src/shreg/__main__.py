"""Command-line interface for shape regularization.

Usage:
    shreg              # Run with visualization
    shreg --no-plot    # Run without visualization (batch mode)
    python -m shreg    # Same as above
"""
from __future__ import annotations

import argparse

from . import plotting
from .contours_reg import load_polylines, regularize_contour
from .segments_reg import (
    create_cgal_example,
    create_example_angles,
    create_example_offsets,
    process_real,
    seg,
    solve_line_segments,
)


def run_segment_examples() -> None:
    """Run all segment regularization examples."""
    print("=" * 60)
    print("SEGMENT REGULARIZATION EXAMPLES")
    print("=" * 60)

    # Example 1: Real data from file
    print("\n[1] Real Data (from real_data_2.xyzi)")
    with plotting.BeforeAfter("Real Data") as plot:
        segments = process_real()
        plot.before(segments)
        segments = solve_line_segments(
            segments, offset=True, angle=True, maximum_angle=80, maximum_offset=2.0
        )
        plot.after(segments)
    print(f"    Regularized {len(segments)} segments")

    # Example 2: Offset regularization (radial pattern)
    print("\n[2] Offset Regularization (radial pattern)")
    with plotting.BeforeAfter("Offset Example") as plot:
        segments = create_example_offsets()
        plot.before(segments)
        segments = solve_line_segments(
            segments, offset=True, angle=False, maximum_offset=0.5
        )
        plot.after(segments)
    print(f"    Regularized {len(segments)} segments")

    # Example 3: Angle regularization (crossing lines)
    print("\n[3] Angle Regularization (crossing lines)")
    with plotting.BeforeAfter("Angle Example") as plot:
        segments = create_example_angles()
        plot.before(segments)
        segments = solve_line_segments(
            segments, offset=False, angle=True, maximum_angle=40
        )
        plot.after(segments)
    print(f"    Regularized {len(segments)} segments")

    # Example 4: CGAL 2.4 example (grouped regularization)
    print("\n[4] CGAL 2.4 Example (3 groups: boundary, top rhombus, bottom rhombus)")
    segments, groups = create_cgal_example()
    group_names = ["Outer Boundary", "Top Rhombus", "Bottom Rhombus"]

    for name, group in zip(group_names, groups):
        with plotting.BeforeAfter(f"CGAL: {name}") as plot:
            subset = [segments[j] for j in group]
            plot.before(subset)
            subset = solve_line_segments(
                subset, offset=True, angle=True, maximum_offset=0.1, maximum_angle=10
            )
            plot.after(subset)
        print(f"    {name}: {len(subset)} segments")

    # Example 5: Hexagon
    print("\n[5] Hexagon (combined angle + offset)")
    with plotting.BeforeAfter("Hexagon") as plot:
        segments = [
            seg(0.2, 0.0, 1.2, 0.0),
            seg(1.2, 0.1, 2.2, 0.1),
            seg(2.2, 0.0, 2.0, 2.0),
            seg(2.0, 2.0, 1.0, 2.0),
            seg(1.0, 1.9, 0.0, 1.9),
            seg(0.0, 2.0, 0.2, 0.0),
        ]
        plot.before(segments)
        segments = solve_line_segments(
            segments, offset=True, angle=True, maximum_offset=0.5, maximum_angle=25
        )
        plot.after(segments)
    print(f"    Regularized {len(segments)} segments")

    # Example 6: Large coordinates
    print("\n[6] Large Coordinates (UTM-like)")
    with plotting.BeforeAfter("Large Coordinates") as plot:
        segments = [
            seg(458255.310417501, 5463361.494450074, 458255.6298964034, 5463357.52230541),
            seg(458256.1043923136, 5463349.528498339, 458256.4568710004, 5463345.146060798),
            seg(458259.4567598508, 5463345.292766673, 458259.7838926231, 5463341.225459929),
        ]
        plot.before(segments)
        segments = solve_line_segments(
            segments, offset=True, angle=False, maximum_offset=4.0
        )
        plot.after(segments)
    print(f"    Regularized {len(segments)} segments")

    # Example 7: Two segments (edge case)
    print("\n[7] Two Segments (edge case)")
    with plotting.BeforeAfter("Two Segments") as plot:
        segments = [
            seg(0.5, -1, 0.5, 0),
            seg(0.0, 0.0, 0.0, 1.0),
        ]
        plot.before(segments)
        segments = solve_line_segments(
            segments, offset=True, angle=False, maximum_offset=0.5
        )
        plot.after(segments)
    print(f"    Regularized {len(segments)} segments")


def run_contour_examples() -> None:
    """Run all contour regularization examples."""
    print("\n" + "=" * 60)
    print("CONTOUR REGULARIZATION EXAMPLES")
    print("=" * 60)

    # Example 1: Simple polygon
    print("\n[1] Simple Polygon (axis alignment)")
    points = [
        [45, 29], [65, 440], [44, 498], [446, 498], [429, 325],
        [499, 309], [448, 206], [479, 148], [479, 31], [247, 88],
    ]
    result = regularize_contour(points, "axis", max_offset=20, visualize=plotting.ENABLED)
    print(f"    Input: {len(points)} points -> Output: {len(result)} points")

    # Example 2: Complex polygon
    print("\n[2] Complex Polygon (longest edge alignment)")
    points = [
        [224, 429], [272, 338], [288, 330], [349, 224], [384, 323],
        [376, 382], [323, 382], [308, 483], [377, 507], [448, 505],
        [465, 415], [451, 381], [468, 300], [425, 172], [379, 184],
        [369, 175], [402, 108], [361, 91], [266, 283], [251, 291],
        [247, 279], [347, 85], [338, 73], [244, 25], [160, 192],
        [136, 209], [139, 223], [98, 217], [41, 330],
    ]
    result = regularize_contour(points, "longest", max_offset=20, visualize=plotting.ENABLED)
    print(f"    Input: {len(points)} points -> Output: {len(result)} points")

    # Example 3: Rectangle-like shape
    print("\n[3] Rectangle-like Shape (axis alignment)")
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
    result = regularize_contour(points, "axis", max_offset=20, visualize=plotting.ENABLED)
    print(f"    Input: {len(points)} points -> Output: {len(result)} points")

    # Example 4: H-shape
    print("\n[4] H-Shape (axis alignment)")
    points = [
        [93, 78], [91, 403], [268, 402], [266, 458], [91, 458],
        [91, 730], [207, 733], [208, 839], [384, 837], [387, 570],
        [327, 567], [327, 516], [386, 514], [387, 404], [443, 410],
        [441, 513], [499, 515], [499, 568], [443, 568], [439, 838],
        [674, 840], [671, 781], [732, 786], [732, 840], [909, 844],
        [911, 569], [733, 565], [732, 623], [675, 624], [674, 515],
        [908, 513], [908, 241], [674, 239], [673, 182], [909, 188],
        [909, 79], [325, 78], [324, 187], [559, 185], [558, 241],
        [268, 241], [270, 78],
    ]
    result = regularize_contour(points, "axis", max_offset=20, visualize=plotting.ENABLED)
    print(f"    Input: {len(points)} points -> Output: {len(result)} points")

    # Example 5: Real data from polylines file
    print("\n[5] Real Contour Data (contour.polylines)")
    try:
        points = load_polylines()
        result = regularize_contour(points, "longest", max_offset=20, visualize=plotting.ENABLED)
        print(f"    Input: {len(points)} points -> Output: {len(result)} points")
    except FileNotFoundError:
        print("    contour.polylines not found, skipping")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Shape Regularization Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    shreg              Run all examples with visualization
    shreg --no-plot    Run all examples without visualization
    shreg --segments   Run only segment regularization examples
    shreg --contours   Run only contour regularization examples

Reference: https://doc.cgal.org/latest/Shape_regularization/
        """,
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable visualization (batch mode)"
    )
    parser.add_argument(
        "--segments", "--segments-only", action="store_true",
        help="Run only segment examples"
    )
    parser.add_argument(
        "--contours", "--contours-only", action="store_true",
        help="Run only contour examples"
    )
    args = parser.parse_args()

    if args.no_plot:
        plotting.disable()

    print("Shape Regularization Demo")
    print("Reference: https://doc.cgal.org/latest/Shape_regularization/")

    # If neither flag is set, run both
    run_segments = not args.contours or args.segments
    run_contours = not args.segments or args.contours

    if run_segments:
        run_segment_examples()

    if run_contours:
        run_contour_examples()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
