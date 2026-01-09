#!/usr/bin/env python3
"""Generate example plots for the README."""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from shreg import (
    solve_line_segments,
    snap_regularize_segments,
    regularize_contour,
    process_real,
    create_example_angles,
    create_cgal_example,
    metric_regularize_segments,
    seg,
    plotting,
)

# Disable interactive plotting
plotting.disable()

# Common plot settings for clean README images
PLOT_OPTS = {
    "show": False,
    "show_titles": False,
    "show_ticks": False,
    "show_grid": True,  # Keep grid for reference
}


def main():
    print("Generating README plots...")

    # =========================================================================
    # Segment Regularization Examples
    # =========================================================================

    # 1. Segment regularization - Real data
    print("  Generating segment_real_data.png...")
    segments = process_real()
    before = [s.copy() for s in segments]
    after = solve_line_segments(
        segments, offset=True, angle=True, maximum_angle=80, maximum_offset=2.0
    )
    plotting.comparison(before, after, save_path="docs/images/segment_real_data.png", **PLOT_OPTS)

    # 2. Angle regularization
    print("  Generating segment_angle.png...")
    segments = create_example_angles()
    before = [s.copy() for s in segments]
    after = solve_line_segments(segments, offset=False, angle=True, maximum_angle=40)
    plotting.comparison(before, after, save_path="docs/images/segment_angle.png", **PLOT_OPTS)

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
    plotting.comparison(before, after, save_path="docs/images/segment_hexagon.png", **PLOT_OPTS)

    # 4. CGAL 2.4 Angle + Offset Regularization example
    print("  Generating segment_cgal_example.png...")
    segments, _groups = create_cgal_example()
    before = [s.copy() for s in segments]
    after = solve_line_segments(
        segments, offset=True, angle=True, maximum_offset=0.1, maximum_angle=10
    )
    plotting.comparison(before, after, save_path="docs/images/segment_cgal_example.png", **PLOT_OPTS)

    # =========================================================================
    # Snap Regularization Examples
    # =========================================================================

    # 5. Snap - Cluster method
    print("  Generating snap_cluster.png...")
    segments = [
        seg(0.0, 0.0, 1.0, 0.05),
        seg(1.08, 0.0, 1.05, 1.0),
        seg(1.0, 1.08, 0.0, 0.95),
        seg(-0.05, 1.0, 0.0, 0.0),
    ]
    before = [s.copy() for s in segments]
    after = snap_regularize_segments(segments, epsilon=0.15, method="cluster")
    plotting.comparison(before, after, save_path="docs/images/snap_cluster.png", **PLOT_OPTS)

    # 6. Snap - Hard constraints
    print("  Generating snap_hard.png...")
    segments = [
        seg(0.0, 0.0, 1.0, 0.05),
        seg(1.08, 0.0, 1.05, 1.0),
        seg(1.0, 1.08, 0.0, 0.95),
        seg(-0.05, 1.0, 0.0, 0.0),
    ]
    before = [s.copy() for s in segments]
    after = snap_regularize_segments(segments, epsilon=0.15, method="hard")
    plotting.comparison(before, after, save_path="docs/images/snap_hard.png", **PLOT_OPTS)

    # 7. Snap - Soft constraints
    print("  Generating snap_soft.png...")
    segments = [
        seg(0.0, 0.0, 1.0, 0.05),
        seg(1.08, 0.0, 1.05, 1.0),
        seg(1.0, 1.08, 0.0, 0.95),
        seg(-0.05, 1.0, 0.0, 0.0),
    ]
    before = [s.copy() for s in segments]
    after = snap_regularize_segments(segments, epsilon=0.25, method="soft", soft_weight=50.0)
    plotting.comparison(before, after, save_path="docs/images/snap_soft.png", **PLOT_OPTS)

    # 8. Snap - T-Junction
    print("  Generating snap_tjunction.png...")
    segments = [
        seg(0.0, 0.0, 2.0, 0.0),
        seg(0.0, 1.0, 2.0, 1.0),
        seg(0.95, -0.08, 1.05, 1.1),
    ]
    before = [s.copy() for s in segments]
    after = snap_regularize_segments(segments, epsilon=0.15, method="cluster", t_junctions=True)
    plotting.comparison(before, after, save_path="docs/images/snap_tjunction.png", **PLOT_OPTS)

    # 9. Snap - Complex polygon
    print("  Generating snap_complex.png...")
    segments = [
        seg(0.0, 0.0, 2.0, 0.05),
        seg(2.08, 0.0, 3.0, 1.0),
        seg(2.95, 1.08, 2.0, 2.0),
        seg(2.0, 1.95, 1.0, 2.05),
        seg(0.95, 2.0, 0.0, 1.0),
        seg(0.05, 0.95, 0.0, 0.0),
    ]
    before = [s.copy() for s in segments]
    after = snap_regularize_segments(segments, epsilon=0.15, method="cluster")
    plotting.comparison(before, after, save_path="docs/images/snap_complex.png", **PLOT_OPTS)

    # =========================================================================
    # Contour Regularization Examples
    # =========================================================================

    # 10. Contour regularization - Simple polygon
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
        save_path="docs/images/contour_simple.png",
        **PLOT_OPTS,
        show_point_counts=False,
    )

    # 11. Contour regularization - Rectangle-like shape
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
        save_path="docs/images/contour_rectangle.png",
        **PLOT_OPTS,
        show_point_counts=False,
    )

    # =========================================================================
    # Metric & Pattern Regularization Examples
    # =========================================================================

    # Settings for metric plots (with length annotations)
    METRIC_OPTS = {
        "show": False,
        "show_titles": False,
        "show_ticks": False,
        "show_grid": True,
        "show_lengths": True,
    }

    # 12. Equal Length Regularization
    print("  Generating metric_equal_length.png...")
    segments = [
        seg(0.0, 0.0, 2.0, 0.0),    # length 2.0
        seg(0.0, 1.0, 2.15, 1.0),   # length 2.15
        seg(0.0, 2.0, 1.9, 2.0),    # length 1.9
        seg(0.0, 3.0, 2.05, 3.0),   # length 2.05
    ]
    before = [s.copy() for s in segments]
    after = metric_regularize_segments(
        segments,
        equal_length=True,
        length_quantization=False,
        equal_spacing=False,
        length_tolerance=0.15,
    )
    plotting.metric_comparison(before, after, save_path="docs/images/metric_equal_length.png", **METRIC_OPTS)

    # 13. Length Quantization
    print("  Generating metric_quantization.png...")
    segments = [
        seg(0.0, 0.0, 1.85, 0.0),   # length 1.85 -> 2.0
        seg(0.0, 1.0, 3.15, 1.0),   # length 3.15 -> 3.0
        seg(0.0, 2.0, 0.9, 2.0),    # length 0.9 -> 1.0
        seg(0.0, 3.0, 2.2, 3.0),    # length 2.2 -> 2.0
    ]
    before = [s.copy() for s in segments]
    after = metric_regularize_segments(
        segments,
        equal_length=False,
        length_quantization=True,
        equal_spacing=False,
        base_unit=1.0,
        quantization_tolerance=0.3,
    )
    plotting.metric_comparison(before, after, save_path="docs/images/metric_quantization.png", **METRIC_OPTS)

    # 14. Equal Spacing (Parallel Lines) - Custom plot with spacing annotations
    print("  Generating metric_equal_spacing.png...")
    segments = [
        seg(0.0, 0.0, 3.0, 0.0),    # y=0.0
        seg(0.0, 0.9, 3.0, 0.9),    # y=0.9 (should be 1.0)
        seg(0.0, 2.0, 3.0, 2.0),    # y=2.0
        seg(0.0, 3.1, 3.0, 3.1),    # y=3.1 (should be 3.0)
        seg(0.0, 4.0, 3.0, 4.0),    # y=4.0
    ]
    before = [s.copy() for s in segments]
    after = metric_regularize_segments(
        segments,
        equal_length=False,
        length_quantization=False,
        equal_spacing=True,
        angle_tolerance=5.0,
    )
    # Custom plot with spacing annotations
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Calculate bounds and make square
    all_segs = before + after
    all_x = [s[0] for s in all_segs] + [s[2] for s in all_segs]
    all_y = [s[1] for s in all_segs] + [s[3] for s in all_segs]
    margin = 0.15
    xmin, xmax = min(all_x) - margin * (max(all_x) - min(all_x) or 1), max(all_x) + margin * (max(all_x) - min(all_x) or 1)
    ymin, ymax = min(all_y) - margin * (max(all_y) - min(all_y) or 1), max(all_y) + margin * (max(all_y) - min(all_y) or 1)
    # Make square
    x_range, y_range = xmax - xmin, ymax - ymin
    if x_range > y_range:
        diff = (x_range - y_range) / 2
        ymin, ymax = ymin - diff, ymax + diff
    else:
        diff = (y_range - x_range) / 2
        xmin, xmax = xmin - diff, xmax + diff

    def draw_with_spacing(ax, segs, color):
        # Sort segments by y position
        sorted_segs = sorted(segs, key=lambda s: (s[1] + s[3]) / 2)

        # Draw segments
        for s in sorted_segs:
            ax.plot([s[0], s[2]], [s[1], s[3]], color=color, linewidth=1.5)

        # Draw spacing annotations between consecutive segments
        for i in range(len(sorted_segs) - 1):
            y1 = (sorted_segs[i][1] + sorted_segs[i][3]) / 2
            y2 = (sorted_segs[i+1][1] + sorted_segs[i+1][3]) / 2
            spacing = y2 - y1

            # Position annotation to the right of segments
            x_pos = max(sorted_segs[i][0], sorted_segs[i][2]) + 0.3
            y_mid = (y1 + y2) / 2

            # Draw bracket/arrow
            ax.annotate('', xy=(x_pos, y2), xytext=(x_pos, y1),
                       arrowprops=dict(arrowstyle='<->', color=color, lw=1))
            ax.text(x_pos + 0.15, y_mid, f'{spacing:.2f}', fontsize=8,
                   va='center', ha='left', color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85))

    draw_with_spacing(ax1, before, '#2C3E50')
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    draw_with_spacing(ax2, after, '#27AE60')
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    fig.savefig("docs/images/metric_equal_spacing.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Done! Plots saved to docs/images/")


if __name__ == "__main__":
    main()
