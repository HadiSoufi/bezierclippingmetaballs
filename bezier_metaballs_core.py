"""
Core algorithm for the Bezier-clipped metaballs lab.

This module contains three separate things, deliberately kept apart:

  1. A faithful transcription of BezierClippedMetaballs.py (the proof of concept).
     Same arrays, same numpy calls, same arithmetic, same order of operations.
     Every place it had to be changed to run at all is marked `# DEVIATION:` and
     listed in DEVIATIONS so the UI can show them.

  2. Instrumentation. `trace_ray` records every intermediate value so the UI can
     draw what the algorithm actually did, rather than re-deriving it for display.

  3. Ground truth. Direct evaluation of Wyvill's field function summed over balls.
     This is not a competing implementation of the algorithm -- it is the
     definition of the surface, used as a measuring instrument.

Reference: Nishita & Nakamae, "A Method for Displaying Metaballs by using Bezier
Clipping" (see `bezier clipped metaballs relevant part.docx` in this folder).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Repairs that were required to make the PoC executable at all.
# --------------------------------------------------------------------------

DEVIATIONS = [
    ("BezierClippedMetaballs.py:134",
     "r.Metaballs.copy().extend(s.Metaballs) evaluates to None, so the range "
     "carries no balls and len(r.Metaballs) raises TypeError.",
     "Build the combined list explicitly."),
    ("BezierClippedMetaballs.py:178",
     "r.clip does not exist on Range (the attribute is clip_point); "
     "AttributeError whenever a partial overlap is reached.",
     "Use r.clip_point."),
    ("BezierClippedMetaballs.py:113",
     "math.sqrt(det) raises ValueError for any ball the ray misses. The PoC's "
     "hardcoded scene happened to have both balls hit.",
     "Skip the ball when det < 0, matching Metaballs.hlsl:56."),
    ("BezierClippedMetaballs.py:107",
     "fc = center * -1 hardcodes the ray origin to the world origin, so the "
     "ray cannot be moved.",
     "fc = origin - center, matching Metaballs.hlsl:48."),
    ("BezierClippedMetaballs.py:208",
     "T = t_min / t_max divides by zero once t_max collapses to 0.",
     "Guard with div(); T becomes 0 in that case."),
    ("BezierClippedMetaballs.py:76",
     "normalize_curve_x divides by (curve[6][0] - curve[0][0]), which is 0 "
     "after a degenerate clip, producing nan/inf that then poison everything.",
     "Guard the divide and leave the curve untouched when the span is 0."),
    ("BezierClippedMetaballs.py:158/161",
     "(r.start - ball.start) / ball.length() divides by zero for a ray exactly "
     "tangent to a ball, where the chord has no length.",
     "Route through div()."),
]


# --------------------------------------------------------------------------
# Bugs found in the PoC that are FAITHFULLY PRESERVED (not repaired).
# These are what the harness exists to make visible.
# --------------------------------------------------------------------------

OBSERVATIONS = [
    ("O1", "BezierClippedMetaballs.py:25-26",
     "a is computed as (det/R)^2. det here is the discriminant, which equals "
     "the squared chord length L^2, so this evaluates to L^4/R^2. The paper "
     "(eq. 10) and Metaballs.hlsl:172 need a = (L/2R)^2. Off by a square and "
     "by a factor of 16.",
     "Variant switch: A_MODE"),
    ("O2", "BezierClippedMetaballs.py:172",
     "In the multi-ball branch, d is re-initialised to the zero curve INSIDE "
     "the `for ball in r.Metaballs` loop, so each ball wipes out the previous "
     "one's contribution. Only the last ball survives. The whole point of the "
     "method is that overlapping balls composite by summing control points.",
     None),
    ("O3", "BezierClippedMetaballs.py:133",
     "The middle (overlap) range is constructed with clip_point = 0, so its "
     "clip_normal() is (0 - start)/length -- an arbitrary negative number "
     "unrelated to where the clip should happen.",
     None),
    ("O4", "BezierClippedMetaballs.py:161-162",
     "For the left part of a partial overlap, clip_normal() is 1, so the "
     "T == 1 branch runs and uses (ball.end - r.end)/ball.length(). After the "
     "swap on line 137, r.end is the OTHER ball's start, so this is the "
     "overlap fraction, not the fraction of this ball's curve to keep.",
     None),
    ("O5", "BezierClippedMetaballs.py:247",
     "surface = g_offset + range_start adds a normalized curve parameter "
     "(0..1) to a world-space distance. Every clip + normalize_curve_x pair "
     "rescales that parameter, so g_offset must be mapped back through the "
     "accumulated affine transform to become a distance. Note this is not "
     "just 'times the range length': a non-overlapped ball is clipped to "
     "T = 0.5 first (see O10), so the curve spans only half the range.",
     "Variant switch: SURFACE_MODE"),
    ("O6", "BezierClippedMetaballs.py:218-221",
     "The hull slope accumulators are seeded at 0 instead of +/-inf, so a "
     "slope of exactly 0 is always forced into both the min and the max. "
     "Metaballs.hlsl:506-509 seeds them at +/-1e10 instead.",
     "Variant switch: HULL_MODE"),
    ("O7", "BezierClippedMetaballs.py:238-239",
     "t_min = max(start_min_theta_x, start_max_theta_x) takes the FARTHER of "
     "the two candidate crossings. Convex-hull clipping needs the nearer one "
     "for a safe lower bound; taking the farther one can step past the root.",
     "Variant switch: HULL_MODE"),
    ("O8", "BezierClippedMetaballs.py:125",
     "`ranges` is never sorted by start distance -- balls are processed in "
     "input order. Overlap detection only compares ranges[i] with "
     "ranges[i+1], so results depend on the order balls were declared. "
     "Metaballs.hlsl:74-77 insertion-sorts them.",
     None),
    ("O9", "BezierClippedMetaballs.py:131-140",
     "Overlaps are only ever detected between adjacent pairs, and the pair "
     "split is applied in a single forward pass. Three mutually overlapping "
     "balls cannot be represented. All three drafts share this limit.",
     None),
    ("O11", "BezierClippedMetaballs.py:115",
     "start = min(t0, t1) with no clamp, so when the ray origin is INSIDE a "
     "ball the range starts behind the origin and the search can return a "
     "negative surface distance. Metaballs.hlsl:63 guards this with "
     "`not_in_sphere * min(t0, t1)`, zeroing the near root when c <= 0. The "
     "PoC has no equivalent, which is invisible in its hardcoded scene "
     "because the origin sits outside both balls.",
     None),
    ("O10", "BezierClippedMetaballs.py:121 + 155-156",
     "NOT A BUG -- worth knowing. A non-overlapped range gets clip_point = "
     "(start+end)/2, so clip_normal() is exactly 0.5 and the curve is clipped "
     "to its left half. That is deliberate: the density curve is symmetric "
     "and rises to its peak at the middle, so the left half holds exactly one "
     "root, which is the nearest one -- all an opaque surface needs. "
     "BezierClippedMetaballs.h:204 hardcodes the same 0.5. The consequence is "
     "the parameter/world scale factor in O5.",
     None),
]


# --------------------------------------------------------------------------
# Variant switches. Default is always the PoC's own behaviour.
# --------------------------------------------------------------------------

A_POC = "poc"            # a = (det / R)^2
A_PAPER = "paper"        # a = (L / 2R)^2

HULL_POC = "poc"         # seed slopes at 0, t_min = max(...), t_max = min(...)
HULL_HLSL = "hlsl"       # signed-branch derivation from Metaballs.hlsl:531-557

SURFACE_POC = "poc"      # surface = g_offset + range_start
SURFACE_SCALED = "scaled"  # surface = range_start + g_offset * range_length


@dataclass
class Variants:
    a_mode: str = A_POC
    hull_mode: str = HULL_POC
    surface_mode: str = SURFACE_POC


# ==========================================================================
# 1. FAITHFUL TRANSCRIPTION OF BezierClippedMetaballs.py
# ==========================================================================

class Metaball:
    def __init__(self, position, radius, start, end, det):
        # Ensure position is a 2D vector (2 elements)
        if len(position) != 2:
            raise ValueError("Position must be a 2-vector.")
        self.position = np.array(position)  # Store position as a NumPy array
        self.radius = float(radius)  # Radius as a float
        self.start = float(start)
        self.end = float(end)
        self.det = float(det)  # Determinant as a float

    def __repr__(self):
        return f"Metaball(radius={self.radius})"

    def length(self):
        return abs(self.end - self.start)

    def a_value(self, a_mode=A_POC):
        """The `c` local in the PoC's density_control_points -- the paper's a_i."""
        if a_mode == A_PAPER:
            # a = (L / 2R)^2, per eq. (10) and Metaballs.hlsl:172
            c = self.length() / (2.0 * self.radius)
            c *= c
        else:
            # PoC as written: c = det / radius, then squared.
            c = self.det / self.radius
            c *= c
        return c

    def density_control_points(self, a_mode=A_POC):
        c = self.a_value(a_mode)
        points = np.array([[0, 0], [1 / 6, 0], [2 / 6, 0], [3 / 6, 0],
                           [4 / 6, 0], [5 / 6, 0], [1, 0]], dtype=float)
        points[2][1] = 16 / 27 * c ** 2
        points[3][1] = 8 * c ** 2 * (8 * c + 5) / 45
        points[4][1] = 16 / 27 * c ** 2
        return points


class Range:
    def __init__(self, start, end, clip_point, clip_left, Metaballs):
        self.start = float(start)
        self.end = float(end)
        self.clip_point = clip_point
        self.clip_left = clip_left
        self.Metaballs = Metaballs

    def __repr__(self):
        return f"Range(start={self.start}, end={self.end})"

    def length(self):
        return abs(self.end - self.start)

    def clip_normal(self):
        return div(self.clip_point - self.start, self.length())


# Keeps the left half
def clip_curve_left(curve, point):
    clipped = curve.copy()
    clipped[0] = curve[0]
    clipped[1] = point * curve[1] + (1 - point) * curve[0]
    clipped[2] = point**2 * curve[2] + point * (2 - 2 * point) * curve[1] + (1 - point)**2 * curve[0]
    clipped[3] = point**3 * curve[3] + point**2 * (3 - 3 * point) * curve[2] + 3 * point * (1 - point)**2 * curve[1] + (1 - point)**3 * curve[0]
    clipped[4] = point**4 * curve[4] + point**3 * (4 - 4 * point) * curve[3] + 6 * point**2 * (1 - point)**2 * curve[2] + 4 * point * (1 - point)**3 * curve[1] + (1 - point)**4 * curve[0]
    clipped[5] = point**5 * curve[5] + point**4 * (5 - 5 * point) * curve[4] + 10 * point**3 * (1 - point)**2 * curve[3] + 10 * point**2 * (1 - point)**3 * curve[2] + 5 * point * (1 - point)**4 * curve[1] + (1 - point)**5 * curve[0]
    clipped[6] = point**6 * curve[6] + point**5 * (6 - 6 * point) * curve[5] + 15 * point**4 * (1 - point)**2 * curve[4] + 20 * point**3 * (1 - point)**3 * curve[3] + 15 * point**2 * (1 - point)**4 * curve[2] + 6 * point * (1 - point)**5 * curve[1] + (1 - point)**6 * curve[0]
    return clipped


# Keeps the right half
def clip_curve_right(curve, point):
    clipped = curve.copy()
    clipped[6] = curve[6]
    clipped[5] = point * curve[6] + (1 - point) * curve[5]
    clipped[4] = point**2 * curve[6] + point * 2 * (1 - point) * curve[5] + (1 - point)**2 * curve[4]
    clipped[3] = point**3 * curve[6] + point**2 * 3 * (1 - point) * curve[5] + 3 * point * (1 - point)**2 * curve[4] + (1 - point)**3 * curve[3]
    clipped[2] = point**4 * curve[6] + point**3 * 4 * (1 - point) * curve[5] + 6 * point**2 * (1 - point)**2 * curve[4] + 4 * point * (1 - point)**3 * curve[3] + (1 - point)**4 * curve[2]
    clipped[1] = point**5 * curve[6] + point**4 * 5 * (1 - point) * curve[5] + 10 * point**3 * (1 - point)**2 * curve[4] + 10 * point**2 * (1 - point)**3 * curve[3] + 5 * point * (1 - point)**4 * curve[2] + (1 - point)**5 * curve[1]
    clipped[0] = point**6 * curve[6] + point**5 * 6 * (1 - point) * curve[5] + 15 * point**4 * (1 - point)**2 * curve[4] + 20 * point**3 * (1 - point)**3 * curve[3] + 15 * point**2 * (1 - point)**4 * curve[2] + 6 * point * (1 - point)**5 * curve[1] + (1 - point)**6 * curve[0]
    return clipped


# Make all x values fall between 0 and 1
def normalize_curve_x(curve):
    dist = curve[6][0] - curve[0][0]
    offset = curve[0][0]
    # DEVIATION: guard the zero-width span (see DEVIATIONS).
    if dist == 0:
        return curve.copy()
    normalized = [[(p[0] - offset) / dist, p[1]] for p in curve.copy()]
    return np.array(normalized)


# Does given curve reach a y-value
def curve_below(curve, y):
    max_y = np.max(curve[:, 1])
    return max_y < y


def div(a, b):
    return a / b if b else 0


# ==========================================================================
# 2. INSTRUMENTED DRIVER
# ==========================================================================

@dataclass
class BallRecord:
    index: int
    center: Tuple[float, float]
    radius: float
    det: float
    t0: float
    t1: float
    start: float
    end: float
    a: float
    control_y: List[float]
    missed: bool = False


@dataclass
class RangeRecord:
    start: float
    end: float
    clip_point: float
    clip_left: bool
    ball_count: int
    ball_indices: List[int]
    label: str = ""


@dataclass
class IterRecord:
    i: int
    curve: np.ndarray          # 7x2, after this iteration's clip + normalize
    start_min_theta: float
    start_max_theta: float
    end_min_theta: float
    end_max_theta: float
    t_min: float
    t_max: float
    g_offset: float
    g_ratio: float


@dataclass
class Trace:
    origin: np.ndarray
    direction: np.ndarray
    threshold: float
    variants: Variants

    balls: List[BallRecord] = dc_field(default_factory=list)
    ranges_initial: List[RangeRecord] = dc_field(default_factory=list)
    ranges_split: List[RangeRecord] = dc_field(default_factory=list)

    found_surface: bool = False
    selected_range: Optional[RangeRecord] = None
    selected_index: int = -1
    range_start: float = 0.0
    range_length: float = 0.0
    selected_curve: Optional[np.ndarray] = None

    # World-space span the selected (already clipped + normalized) curve covers.
    # Curve parameter p in [0,1] maps to distance curve_t0 + p*(curve_t1-curve_t0).
    curve_t0: float = 0.0
    curve_t1: float = 0.0

    iters: List[IterRecord] = dc_field(default_factory=list)
    g_offset: float = 0.0
    g_ratio: float = 1.0
    surface: Optional[float] = None
    hit_point: Optional[np.ndarray] = None

    notes: List[str] = dc_field(default_factory=list)

    @property
    def ranges_sorted(self) -> bool:
        s = [r.start for r in self.ranges_initial]
        return all(s[i] <= s[i + 1] for i in range(len(s) - 1))


def _hull_bounds_poc(d, Threshold):
    """BezierClippedMetaballs.py:214-239, verbatim."""
    start = d[0]
    end = d[6]

    start_max_theta = 0
    start_min_theta = 0
    end_max_theta = 0
    end_min_theta = 0

    for j, p in enumerate(d):
        theta = div(p[1] - start[1], p[0] - start[0])
        start_min_theta = min(start_min_theta, theta)
        start_max_theta = max(start_max_theta, theta)

        theta = div(p[1] - end[1], p[0] - end[0])
        end_min_theta = min(end_min_theta, theta)
        end_max_theta = max(end_max_theta, theta)

    start_min_theta_x = div(Threshold - start[1], start_min_theta) + start[0]
    start_max_theta_x = div(Threshold - start[1], start_max_theta) + start[0]

    end_min_theta_x = div(Threshold - end[1], end_min_theta) + end[0]
    end_max_theta_x = div(Threshold - end[1], end_max_theta) + end[0]

    t_min = max(start_min_theta_x, start_max_theta_x)
    t_max = min(end_min_theta_x, end_max_theta_x)

    return (t_min, t_max, start_min_theta, start_max_theta,
            end_min_theta, end_max_theta)


def _hull_bounds_hlsl(d, Threshold):
    """Metaballs.hlsl:506-560, the signed-branch derivation."""
    start_min_theta = 1e10
    start_max_theta = -1e10
    end_min_theta = 1e10
    end_max_theta = -1e10

    for j in range(1, 7):
        dx_start = d[j][0] - d[0][0]
        if dx_start != 0:
            theta_start = (d[j][1] - d[0][1]) / dx_start
            start_min_theta = min(start_min_theta, theta_start)
            start_max_theta = max(start_max_theta, theta_start)

        dx_end = d[6][0] - d[j - 1][0]
        if dx_end != 0:
            theta_end = (d[6][1] - d[j - 1][1]) / dx_end
            end_min_theta = min(end_min_theta, theta_end)
            end_max_theta = max(end_max_theta, theta_end)

    dy_start = Threshold - d[0][1]
    t_min = 0.0
    if abs(dy_start) > 1e-5:
        if dy_start > 0.0:
            t_min = (dy_start / start_max_theta) if start_max_theta > 1e-5 else 1.0
        else:
            t_min = (dy_start / start_min_theta) if start_min_theta < -1e-5 else 1.0

    dy_end = Threshold - d[6][1]
    t_max = 1.0
    if abs(dy_end) > 1e-5:
        if dy_end > 0.0:
            t_max = (1.0 + dy_end / end_min_theta) if end_min_theta < -1e-5 else 0.0
        else:
            t_max = (1.0 + dy_end / end_max_theta) if end_max_theta > 1e-5 else 0.0

    t_min = min(max(t_min, 0.0), 1.0)
    t_max = min(max(t_max, 0.0), 1.0)

    return (t_min, t_max, start_min_theta, start_max_theta,
            end_min_theta, end_max_theta)


def trace_ray(spheres, origin, direction, Threshold=0.2, max_iters=20,
              tol=0.001, variants: Optional[Variants] = None,
              record: bool = True) -> Trace:
    """Run the PoC's algorithm for one ray.

    `spheres` is a sequence of (cx, cy, radius). `origin` and `direction` are
    2-vectors; `direction` is normalized here.
    """
    if variants is None:
        variants = Variants()

    origin = np.asarray(origin, dtype=float)
    TraceDirection = np.asarray(direction, dtype=float)
    n = np.linalg.norm(TraceDirection)
    if n > 0:
        TraceDirection = TraceDirection / n

    tr = Trace(origin=origin, direction=TraceDirection,
               threshold=Threshold, variants=variants)

    # ---- Load spheres into data structures (PoC lines 103-121) ----
    metaballs = []
    ranges = []

    for si, sphere in enumerate(spheres):
        center = [sphere[0], sphere[1]]
        radius = sphere[2]

        # DEVIATION: PoC had `fc = np.multiply(center, -1)`, pinning the ray
        # origin to (0, 0).
        fc = np.subtract(origin, center)
        b = 2 * np.dot(fc, TraceDirection)
        c = np.dot(fc, fc) - radius * radius
        det = b * b - 4 * c

        # DEVIATION: PoC called math.sqrt(det) unguarded.
        if det < 0:
            if record:
                tr.balls.append(BallRecord(si, (center[0], center[1]), radius,
                                           det, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           [0.0] * 7, missed=True))
            continue

        # Calculate intersections with ray as distances
        t0 = -(b + math.copysign(math.sqrt(det), b)) / 2
        t1 = (t0 != 0) * c / t0
        start = min(t0, t1)
        end = max(t0, t1)

        m = Metaball(center, radius, start, end, det)
        m.source_index = si

        metaballs.append(m)
        ranges.append(Range(start, end, (start + end) / 2, False, [m]))

        if record:
            cps = m.density_control_points(variants.a_mode)
            tr.balls.append(BallRecord(
                si, (center[0], center[1]), radius, det, t0, t1, start, end,
                m.a_value(variants.a_mode), [float(v) for v in cps[:, 1]]))

    if record:
        tr.ranges_initial = [_rr(r, metaballs) for r in ranges]

    if not ranges:
        return tr

    # ---- Break overlapping ranges out into distinct ranges (PoC 124-142) ----
    intersections = []
    for i, r in enumerate(ranges):
        intersections.append(r)

        if i == len(ranges) - 1:
            continue

        s = ranges[i + 1]
        if s.start > r.start and s.start < r.end:
            t = Range(s.start, r.end, 0, False, [None])
            # DEVIATION: `.extend()` returns None in the PoC.
            combined = r.Metaballs.copy()
            combined.extend(s.Metaballs)
            t.Metaballs = combined
            intersections.append(t)

            r.end, s.start = s.start, r.end
            r.clip_point = r.end
            s.clip_point = s.start
            s.clip_left = True

    intersections = sorted(intersections, key=lambda r: r.start)

    if record:
        tr.ranges_split = [_rr(r, metaballs) for r in intersections]
        if not tr.ranges_sorted:
            tr.notes.append(
                "O8: input ranges were not in ascending start order; the PoC "
                "never sorts them, so overlap detection saw a different "
                "pairing than a sorted pass would.")

    # ---- Convert ranges to bezier curves (PoC 145-191) ----
    found_surface = False
    range_start = 0
    d = None
    sel_len = 0.0
    # World span the curve in `d` covers, tracked alongside the PoC's own math
    # purely for reporting -- it never feeds back into the PoC's result.
    span = (0.0, 0.0)

    for ri, r in enumerate(intersections):
        if len(r.Metaballs) == 1:
            ball = r.Metaballs[0]
            d = ball.density_control_points(variants.a_mode)
            T = r.clip_normal()
            bl = ball.length()

            if T != 0 and T != 1:
                d = clip_curve_left(d, T)
                span = (ball.start, ball.start + T * bl)
            elif T == 0:
                T = div(r.start - ball.start, bl)
                d = clip_curve_right(d, T)
                span = (ball.start + T * bl, ball.end)
            elif T == 1:
                T = div(ball.end - r.end, bl)
                d = clip_curve_left(d, T)
                span = (ball.start, ball.start + T * bl)

            d = normalize_curve_x(d)

            if not curve_below(d, Threshold):
                found_surface = True
                range_start = r.start
                sel_len = r.length()
                break
        else:
            for ball in r.Metaballs:
                # NOTE (O2): d is reset here, inside the loop, exactly as the
                # PoC has it. Each ball therefore erases the previous ball's
                # contribution instead of compositing with it.
                d = np.array([[0, 0], [1 / 6, 0], [2 / 6, 0], [3 / 6, 0],
                              [4 / 6, 0], [5 / 6, 0], [1, 0]], dtype=float)

                b = ball.density_control_points(variants.a_mode)
                T = r.clip_normal()
                bl = ball.length()

                if r.clip_left:
                    # DEVIATION: PoC wrote `r.clip`, which does not exist.
                    range_start = r.clip_point + r.start
                    b = clip_curve_left(b, T)
                    span = (ball.start, ball.start + T * bl)
                else:
                    b = clip_curve_right(b, T)
                    span = (ball.start + T * bl, ball.end)

                b = normalize_curve_x(b)

                for i in range(0, 7):
                    d[i][1] += b[i][1]

            if not curve_below(d, Threshold):
                found_surface = True
                range_start = r.start
                sel_len = r.length()
                break

    tr.found_surface = found_surface
    if not found_surface:
        return tr

    if record:
        tr.selected_index = ri
        tr.selected_range = tr.ranges_split[ri] if ri < len(tr.ranges_split) else None
        tr.selected_curve = d.copy()
    tr.range_start = float(range_start)
    tr.range_length = float(sel_len)
    tr.curve_t0, tr.curve_t1 = float(span[0]), float(span[1])

    # ---- Iteratively clip to find the intersection with the threshold ----
    t_max = 1
    t_min = 0
    g_offset = 0
    g_ratio = 1

    for i in range(max_iters):
        # Clip to range. Left first
        T = t_max
        d = clip_curve_left(d, T)

        # Now right of left
        # DEVIATION: PoC wrote `T = t_min / t_max` with no zero guard.
        T = div(t_min, t_max)
        d = clip_curve_right(d, T)

        # Normalize x range
        d = normalize_curve_x(d)

        # gather convex hull
        if variants.hull_mode == HULL_HLSL:
            (t_min, t_max, smin, smax, emin, emax) = _hull_bounds_hlsl(d, Threshold)
        else:
            (t_min, t_max, smin, smax, emin, emax) = _hull_bounds_poc(d, Threshold)

        g_offset += t_min * g_ratio
        g_ratio *= t_max - t_min

        if record:
            tr.iters.append(IterRecord(
                i=i, curve=d.copy(),
                start_min_theta=float(smin), start_max_theta=float(smax),
                end_min_theta=float(emin), end_max_theta=float(emax),
                t_min=float(t_min), t_max=float(t_max),
                g_offset=float(g_offset), g_ratio=float(g_ratio)))

        if t_max - t_min < tol:
            break

    tr.g_offset = float(g_offset)
    tr.g_ratio = float(g_ratio)

    if variants.surface_mode == SURFACE_SCALED:
        # Map the normalized curve parameter back through the affine transform
        # the clips actually applied, instead of adding it to a distance.
        surface = span[0] + g_offset * (span[1] - span[0])
    else:
        surface = g_offset + range_start

    tr.surface = float(surface)
    tr.hit_point = origin + surface * TraceDirection
    return tr


def _rr(r: Range, metaballs) -> RangeRecord:
    idx = []
    for b in r.Metaballs:
        if b is None:
            continue
        idx.append(getattr(b, "source_index", -1))
    return RangeRecord(start=r.start, end=r.end,
                       clip_point=float(r.clip_point),
                       clip_left=bool(r.clip_left),
                       ball_count=len(r.Metaballs),
                       ball_indices=idx)


# ==========================================================================
# 2b. CORRECTED ALGORITHM
#
# Same method as the paper, restructured around what the harness showed to be
# wrong. Written to port straight to HLSL: scalar floats, fixed-size arrays,
# bounded loops, no recursion, no dynamic allocation.
#
# Four changes from the PoC:
#
#   1. a = (L/2R)^2                            (was (det/R)^2)
#   2. A breakpoint sweep replaces pairwise     (was: adjacent pairs only, so
#      overlap classification.                   3+ overlapping balls and any
#                                                non-sorted input broke)
#   3. Curves composite by summing control      (was: d re-initialised inside
#      points over a shared parameter domain.    the per-ball loop, so only the
#                                                last ball survived)
#   4. The root is mapped back to a distance    (was: a normalized parameter
#      through the accumulated affine map.       added to a world distance)
#
# The clipping loop is also specialised to the *first* root, which is all an
# opaque surface needs -- the paper says exactly this in section 5.2. That
# makes it one-sided: only a lower bound on the root is ever needed, so there
# is no t_max, no two-sided clip, and no subdivision-on-multiple-roots case.
# ==========================================================================

def _decasteljau(d, t):
    """Subdivide a degree-6 Bezier at t.

    Returns (left, right): control points of the curve restricted to [0, t]
    and to [t, 1], each already reparameterized onto [0, 1].

    Used in place of the PoC's expanded subdivision polynomials -- same result,
    but 21 lerps instead of 28 expanded terms, and far better conditioned near
    t = 0 and t = 1 where the expanded powers lose precision.
    """
    b = [float(v) for v in d]
    left = [0.0] * 7
    right = [0.0] * 7
    left[0] = b[0]
    right[6] = b[6]
    for lvl in range(1, 7):
        for i in range(7 - lvl):
            b[i] = b[i] + t * (b[i + 1] - b[i])
        left[lvl] = b[0]
        right[6 - lvl] = b[6 - lvl]
    return left, right


def control_points(a):
    """Degree-6 control points for one ball's density over its own chord.

    a = (L / 2R)^2 = 1 - (h/R)^2, with L the chord length and h the distance
    from the ball centre to the ray. Paper eq. (10).
    """
    a2 = a * a
    mid = 16.0 / 27.0 * a2
    return [0.0, 0.0, mid, 8.0 * a2 * (8.0 * a + 5.0) / 45.0, mid, 0.0, 0.0]


def _clip_to(d, u0, u1):
    """Restrict a curve on [0,1] to the sub-interval [u0, u1]."""
    if u0 > 0.0:
        _, d = _decasteljau(d, u0)
        span = 1.0 - u0
        u1 = 1.0 if span <= 0.0 else (u1 - u0) / span
    if u1 < 1.0:
        d, _ = _decasteljau(d, max(u1, 0.0))
    return d


def _lower_hull_crossing(g):
    """First upward zero crossing of the control polygon's LOWER convex hull.

    The curve lies above its lower hull, and the lower hull is convex, so once
    it crosses zero going up it stays up. That makes this crossing a guaranteed
    UPPER bound on the first root -- the other half of the bracket.

    Returns t in (0, 1], or None if the lower hull never reaches zero.
    """
    # Monotone-chain lower hull. x is already increasing (x_k = k/6).
    hx = [0.0] * 7
    hy = [0.0] * 7
    hn = 0
    for k in range(7):
        xk = k / 6.0
        yk = g[k]
        while hn >= 2:
            ax = hx[hn - 1] - hx[hn - 2]
            ay = hy[hn - 1] - hy[hn - 2]
            bx = xk - hx[hn - 2]
            by = yk - hy[hn - 2]
            if ax * by - ay * bx <= 0.0:
                hn -= 1
            else:
                break
        hx[hn] = xk
        hy[hn] = yk
        hn += 1

    for j in range(hn - 1):
        y0 = hy[j]
        y1 = hy[j + 1]
        if y0 < 0.0 <= y1:
            return hx[j] + (hx[j + 1] - hx[j]) * (-y0) / (y1 - y0)
    return None


_BINOM6 = (1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0)


def _eval6(g, t):
    """Bernstein evaluation of a degree-6 Bezier. ~20 flops, no subdivision."""
    s = 1.0 - t
    tk = 1.0
    sk = [1.0] * 7
    for k in range(5, -1, -1):
        sk[k] = sk[k + 1] * s
    acc = 0.0
    for k in range(7):
        acc += _BINOM6[k] * tk * sk[k] * g[k]
        tk *= t
    return acc


def _probe_bracket(g, nprobe=8):
    """Smallest probe point where the curve is already at/above zero.

    The lower hull gives a bracket for free, but only when the curve ends above
    the threshold. A lone ball's density returns to zero at both ends of its
    chord, so its lower hull never crosses and no bracket exists -- which drops
    the search to linear convergence. A short scan finds one directly.

    Scanning in increasing t matters: the first positive probe brackets the
    FIRST root, which is the one an opaque surface needs.
    """
    for i in range(1, nprobe + 1):
        t = i / nprobe
        if _eval6(g, t) >= 0.0:
            return t
    return None


def first_root(g, max_iters=24, tol=1e-9, record=None):
    """Smallest t in [0,1] where a degree-6 Bezier given by `g` reaches 0.

    `g` is the density control points already offset by -Threshold, so the
    problem is "first upward zero crossing".

    Bezier clipping specialised to the first root. Each iteration brackets it:

      lo -- the curve lies BELOW its upper hull, so the upper hull's first zero
            crossing cannot be later than the curve's. The steepest ray from
            the start control point gives it in closed form.
      hi -- the curve lies ABOVE its lower hull, and a convex function that
            crosses zero upward stays up, so the lower hull's first crossing
            cannot be earlier than the curve's.

    Clipping to [lo, hi] shrinks the span from both sides, so the curve on it
    tends to a straight line and both bounds tend to the root quadratically.
    Clipping only the left end (the obvious specialisation) converges merely
    linearly, because the far end keeps the peak in the control polygon and
    the slope bound stays loose -- that costs ~20 iterations instead of ~5.

    Returns t, or None if the curve never reaches 0 on [0,1].
    """
    offset = 0.0
    scale = 1.0

    for _ in range(max_iters):
        if g[0] >= 0.0:
            return offset

        # Steepest ascent from the start point -> lower bound on the root.
        m = 0.0
        for k in range(1, 7):
            s = (g[k] - g[0]) / (k / 6.0)
            if s > m:
                m = s

        # Every control point at or below the first: the whole curve is below
        # zero by the convex hull property, so there is no root.
        if m <= 0.0:
            return None

        lo = -g[0] / m
        if lo >= 1.0:
            return None          # hull reaches zero past the end of the span

        hi = _lower_hull_crossing(g)
        if hi is None or hi <= lo:
            # Free bracket unavailable -- scan for one. Only ever needed on the
            # first iteration: after one two-sided clip the span ends above
            # zero, so the lower hull always crosses from then on.
            hi = _probe_bracket(g)
        if hi is None or hi <= lo:
            hi = 1.0             # still none; clip the left only this round

        if record is not None:
            record.append((list(g), offset, scale, lo, hi, m))

        offset += scale * lo
        scale *= (hi - lo)

        g = _clip_to(g, lo, hi)

        if scale <= tol or (hi - lo) <= tol:
            return offset

    return offset


def trace_ray_fixed(spheres, origin, direction, Threshold=0.2, max_iters=24,
                    tol=1e-9, record: bool = True,
                    variants: Optional[Variants] = None) -> Trace:
    """Corrected ray/metaball intersection. Signature matches `trace_ray`."""
    origin = np.asarray(origin, dtype=float)
    D = np.asarray(direction, dtype=float)
    n = np.linalg.norm(D)
    if n > 0:
        D = D / n

    tr = Trace(origin=origin, direction=D, threshold=Threshold,
               variants=variants or Variants(A_PAPER, HULL_HLSL, SURFACE_SCALED))

    # ---- 1. Ray/sphere intervals -------------------------------------
    b_start = []
    b_end = []
    b_a = []
    b_idx = []

    for i, (cx, cy, R) in enumerate(spheres):
        fx = origin[0] - cx
        fy = origin[1] - cy
        b = 2.0 * (fx * D[0] + fy * D[1])
        c = fx * fx + fy * fy - R * R
        det = b * b - 4.0 * c

        if det < 0.0:
            if record:
                tr.balls.append(BallRecord(i, (cx, cy), R, det, 0.0, 0.0,
                                           0.0, 0.0, 0.0, [0.0] * 7, missed=True))
            continue

        sq = math.sqrt(det)
        t0 = 0.5 * (-b - sq)
        t1 = 0.5 * (-b + sq)

        # Wholly behind the origin.
        if t1 <= 0.0:
            if record:
                tr.balls.append(BallRecord(i, (cx, cy), R, det, t0, t1,
                                           t0, t1, 0.0, [0.0] * 7, missed=True))
            continue

        # The chord is what defines the density curve, so it must stay the
        # geometric chord even when the origin is inside the ball. Clamping
        # `start` to 0 here -- which is what Metaballs.hlsl:63 does -- would
        # silently reshape the curve. The origin-inside case is handled by
        # starting the sweep at t = 0 instead (see below).
        a = (t1 - t0) / (2.0 * R)
        a = a * a

        b_start.append(t0)
        b_end.append(t1)
        b_a.append(a)
        b_idx.append(i)

        if record:
            tr.balls.append(BallRecord(i, (cx, cy), R, det, t0, t1, t0, t1,
                                       a, control_points(a)))

    nb = len(b_start)
    if nb == 0:
        return tr

    # ---- 2. Breakpoints ----------------------------------------------
    # Every chord endpoint, sorted. Between consecutive breakpoints the set of
    # balls covering the ray is constant, which is what makes compositing well
    # defined for any number of overlaps -- no pair classification needed.
    marks = [0.0]
    for k in range(nb):
        if b_start[k] > 0.0:
            marks.append(b_start[k])
        if b_end[k] > 0.0:
            marks.append(b_end[k])

    # Insertion sort: small n, and it is what the shader will do.
    for i in range(1, len(marks)):
        v = marks[i]
        j = i - 1
        while j >= 0 and marks[j] > v:
            marks[j + 1] = marks[j]
            j -= 1
        marks[j + 1] = v

    # ---- 3. Sweep segments, nearest first ----------------------------
    span_eps = 1e-12
    for sgi in range(len(marks) - 1):
        p = marks[sgi]
        q = marks[sgi + 1]
        if q - p <= span_eps:
            continue
        mid = 0.5 * (p + q)

        d = [0.0] * 7
        active = 0
        active_idx = []
        for k in range(nb):
            if b_start[k] <= mid <= b_end[k]:
                L = b_end[k] - b_start[k]
                if L <= 0.0:
                    continue
                u0 = (p - b_start[k]) / L
                u1 = (q - b_start[k]) / L
                ck = _clip_to(control_points(b_a[k]), u0, u1)
                for e in range(7):
                    d[e] += ck[e]
                active += 1
                active_idx.append(b_idx[k])

        if active == 0:
            continue

        if record:
            tr.ranges_split.append(RangeRecord(
                start=p, end=q, clip_point=mid, clip_left=False,
                ball_count=active, ball_indices=active_idx))

        # Convex hull property: if every control point is below the threshold
        # the curve is too, so this segment cannot contain a crossing.
        if max(d) < Threshold:
            continue

        g = [v - Threshold for v in d]
        steps = [] if record else None
        root = first_root(g, max_iters, tol, steps)
        if root is None:
            continue

        tr.found_surface = True
        tr.selected_index = len(tr.ranges_split) - 1 if record else -1
        tr.selected_range = tr.ranges_split[-1] if record else None
        tr.range_start = p
        tr.range_length = q - p
        tr.curve_t0 = p
        tr.curve_t1 = q
        tr.g_offset = root
        tr.g_ratio = 1.0
        tr.surface = p + root * (q - p)
        tr.hit_point = origin + tr.surface * D

        if record:
            tr.selected_curve = np.stack(
                [np.array([k / 6.0 for k in range(7)]), np.array(d)], axis=-1)
            for i2, (gv, off, sc, lo, hi, m) in enumerate(steps):
                tr.iters.append(IterRecord(
                    i=i2,
                    curve=np.stack([np.array([k / 6.0 for k in range(7)]),
                                    np.array(gv) + Threshold], axis=-1),
                    start_min_theta=0.0, start_max_theta=m,
                    end_min_theta=0.0, end_max_theta=0.0,
                    t_min=lo, t_max=hi, g_offset=off, g_ratio=sc))
        return tr

    return tr


# ==========================================================================
# 3. GROUND TRUTH -- Wyvill's degree-6 field, evaluated directly.
# ==========================================================================

def wyvill(u):
    """Wyvill's field as a function of u = (r/R)^2, zero outside the ball.

    f(r) = -4/9 (r/R)^6 + 17/9 (r/R)^4 - 22/9 (r/R)^2 + 1
    """
    u = np.asarray(u, dtype=float)
    f = -4.0 / 9.0 * u**3 + 17.0 / 9.0 * u**2 - 22.0 / 9.0 * u + 1.0
    return np.where(u >= 1.0, 0.0, f)


def field(points, spheres):
    """Sum of Wyvill fields at `points` (..., 2)."""
    p = np.asarray(points, dtype=float)
    total = np.zeros(p.shape[:-1], dtype=float)
    for cx, cy, R in spheres:
        d = p - np.array([cx, cy])
        u = (d[..., 0]**2 + d[..., 1]**2) / (R * R)
        total = total + wyvill(u)
    return total


def field_grad(point, spheres):
    """Analytic gradient of the summed field at a single point."""
    p = np.asarray(point, dtype=float)
    g = np.zeros(2)
    for cx, cy, R in spheres:
        d = p - np.array([cx, cy])
        r2 = d[0]**2 + d[1]**2
        u = r2 / (R * R)
        if u >= 1.0:
            continue
        dfdu = -12.0 / 9.0 * u**2 + 34.0 / 9.0 * u - 22.0 / 9.0
        g = g + dfdu * (2.0 * d / (R * R))
    return g


def field_normal(point, spheres):
    """Outward surface normal: the field decreases outward, so it is -grad."""
    g = field_grad(point, spheres)
    n = np.linalg.norm(g)
    if n == 0:
        return np.array([0.0, 0.0])
    return -g / n


def sample_along_ray(spheres, origin, direction, t0, t1, n=400):
    """Vectorized field sampling along a ray. Returns (t, f)."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) or 1.0)
    t = np.linspace(t0, t1, n)
    pts = origin[None, :] + t[:, None] * direction[None, :]
    return t, field(pts, spheres)


def reference_hit(spheres, origin, direction, threshold=0.2,
                  t_near=0.0, t_far=None, samples=512, bisect_iters=40):
    """First t >= t_near where the summed field crosses `threshold`.

    Dense sample to bracket the first crossing, then bisect. Returns
    (t, point, normal) or (None, None, None).
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) or 1.0)

    if t_far is None:
        t_far = _scene_far(spheres, origin)
    if t_far <= t_near:
        return None, None, None

    t, f = sample_along_ray(spheres, origin, direction, t_near, t_far, samples)
    g = f - threshold

    if g[0] >= 0:
        # Origin already inside the isosurface.
        p = origin + t_near * direction
        return float(t_near), p, field_normal(p, spheres)

    idx = np.nonzero((g[:-1] < 0) & (g[1:] >= 0))[0]
    if idx.size == 0:
        return None, None, None

    lo, hi = t[idx[0]], t[idx[0] + 1]
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        p = origin + mid * direction
        if field(p, spheres) - threshold < 0:
            lo = mid
        else:
            hi = mid
    tm = 0.5 * (lo + hi)
    p = origin + tm * direction
    return float(tm), p, field_normal(p, spheres)


def reference_hit_grid(spheres, origins, direction, threshold=0.2,
                       t_near=0.0, t_far=None, samples=256, bisect_iters=32):
    """Batched form of `reference_hit` for a whole grid of ray origins.

    `origins` is (..., 2), all rays share `direction`. Instead of solving one
    ray at a time, every ray marches in lockstep so each step is a single
    vectorized field evaluation over the whole grid. That turns
    N_rays * N_samples scalar evaluations into N_samples grid evaluations,
    which is what makes a full-frame reference render interactive.

    Returns a float array shaped like origins[..., 0], NaN where the ray misses.
    """
    origins = np.asarray(origins, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) or 1.0)
    shape = origins.shape[:-1]

    if t_far is None:
        t_far = max(_scene_far(spheres, o)
                    for o in origins.reshape(-1, 2)[::max(origins.size // 512, 1)])
    if t_far <= t_near:
        return np.full(shape, np.nan)

    def g_at(t):
        """Field minus threshold at parameter t (scalar or per-ray array)."""
        t = np.asarray(t, dtype=float)
        pts = origins + t[..., None] * direction
        return field(pts, spheres) - threshold

    lo = np.full(shape, np.nan)
    hi = np.full(shape, np.nan)
    found = np.zeros(shape, dtype=bool)

    prev = g_at(np.full(shape, t_near))

    # Rays that start inside the isosurface hit immediately.
    inside = prev >= 0
    if inside.any():
        lo[inside] = t_near
        hi[inside] = t_near
        found |= inside

    ts = np.linspace(t_near, t_far, samples)
    for k in range(1, samples):
        if found.all():
            break
        cur = g_at(np.full(shape, ts[k]))
        cross = (~found) & (prev < 0) & (cur >= 0)
        if cross.any():
            lo[cross] = ts[k - 1]
            hi[cross] = ts[k]
            found |= cross
        prev = cur

    if not found.any():
        return np.full(shape, np.nan)

    # Vector bisection, only on the rays that bracketed a crossing.
    lo_f = lo[found]
    hi_f = hi[found]
    o_f = origins[found]
    for _ in range(bisect_iters):
        mid = 0.5 * (lo_f + hi_f)
        g = field(o_f + mid[..., None] * direction, spheres) - threshold
        neg = g < 0
        lo_f = np.where(neg, mid, lo_f)
        hi_f = np.where(neg, hi_f, mid)

    out = np.full(shape, np.nan)
    out[found] = 0.5 * (lo_f + hi_f)
    return out


def _scene_far(spheres, origin):
    if not len(spheres):
        return 1.0
    far = 0.0
    for cx, cy, R in spheres:
        far = max(far, float(np.hypot(cx - origin[0], cy - origin[1]) + R))
    return far * 1.05 + 1e-6


# ==========================================================================
# 4. Bezier helpers used for plotting and testing
# ==========================================================================

def eval_bezier(ctrl, ts):
    """Evaluate a Bezier curve of degree len(ctrl)-1 at parameters ts."""
    ctrl = np.asarray(ctrl, dtype=float)
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    n = ctrl.shape[0] - 1
    out = np.zeros((ts.shape[0], ctrl.shape[1]))
    for k in range(n + 1):
        b = math.comb(n, k) * ts**k * (1.0 - ts)**(n - k)
        out += b[:, None] * ctrl[k]
    return out


def convex_hull(points):
    """Monotone-chain convex hull; returns the hull in CCW order."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


# Marching-squares case table. Corners are numbered BL=0, BR=1, TR=2, TL=3 and
# the case index is b0 | b1<<1 | b2<<2 | b3<<3 with bk set when that corner is
# above the level. Edges are B=0 (BL-BR), R=1 (BR-TR), T=2 (TL-TR), L=3 (BL-TL).
_MS_TABLE = {
    1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
    5: [(3, 0), (1, 2)],                      # saddle
    6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)], 9: [(2, 0)],
    10: [(0, 1), (2, 3)],                     # saddle
    11: [(2, 1)], 12: [(1, 3)], 13: [(1, 0)], 14: [(0, 3)],
}


def iso_contour(Z, gx, gy, level, chain_tol=1e-9):
    """Vectorized marching squares returning a list of polylines.

    Used instead of matplotlib's contour() for the live scene view: contour()
    costs a fixed ~14 ms per call regardless of grid size (artist construction
    dominates), which is far too slow to run on every mouse-move. This is ~1 ms.

    Z is (ny, nx) sampled on the grid gx (x, length nx) by gy (y, length ny).
    Returns polylines as (N, 2) float arrays.
    """
    Z = np.asarray(Z, dtype=float)
    v0 = Z[:-1, :-1]   # bottom-left
    v1 = Z[:-1, 1:]    # bottom-right
    v2 = Z[1:, 1:]     # top-right
    v3 = Z[1:, :-1]    # top-left

    case = ((v0 > level).astype(np.uint8)
            | ((v1 > level).astype(np.uint8) << 1)
            | ((v2 > level).astype(np.uint8) << 2)
            | ((v3 > level).astype(np.uint8) << 3))

    # Cell corner coordinates by broadcasting, not four meshgrid allocations.
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    x0 = gx[None, :-1]
    x1 = gx[None, 1:]
    y0 = gy[:-1, None]
    y1 = gy[1:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        tb = (level - v0) / (v1 - v0)
        tr = (level - v1) / (v2 - v1)
        tt = (level - v3) / (v2 - v3)
        tl = (level - v0) / (v3 - v0)
    tb, tr, tt, tl = (np.nan_to_num(t, nan=0.5, posinf=0.5, neginf=0.5)
                      for t in (tb, tr, tt, tl))

    shp = case.shape
    bx0 = np.broadcast_to(x0, shp)
    bx1 = np.broadcast_to(x1, shp)
    by0 = np.broadcast_to(y0, shp)
    by1 = np.broadcast_to(y1, shp)
    edges = [
        (x0 + (x1 - x0) * tb, by0),         # 0 bottom
        (bx1, y0 + (y1 - y0) * tr),         # 1 right
        (x0 + (x1 - x0) * tt, by1),         # 2 top
        (bx0, y0 + (y1 - y0) * tl),         # 3 left
    ]

    segs = []
    for c, pairs in _MS_TABLE.items():
        m = case == c
        if not m.any():
            continue
        for ea, eb in pairs:
            ax_, ay_ = edges[ea][0][m], edges[ea][1][m]
            bx_, by_ = edges[eb][0][m], edges[eb][1][m]
            segs.append(np.stack([np.stack([ax_, ay_], axis=-1),
                                  np.stack([bx_, by_], axis=-1)], axis=1))
    if not segs:
        return []
    segs = np.concatenate(segs, axis=0)

    # Chain segments end-to-end so the caller can draw a few long polylines
    # rather than thousands of two-point lines. Endpoint identity is resolved
    # by quantizing to an integer code -- computed once, vectorized, because
    # doing it inside the walk loop dominates the whole function otherwise.
    scale = max(abs(gx[-1] - gx[0]), abs(gy[-1] - gy[0])) or 1.0
    q = 1.0 / max(chain_tol, scale * 1e-7)
    codes = (np.round(segs[:, :, 0] * q).astype(np.int64) * np.int64(73856093)
             ^ np.round(segs[:, :, 1] * q).astype(np.int64) * np.int64(19349663))
    code_list = codes.tolist()

    adj = {}
    for i, (c0, c1) in enumerate(code_list):
        adj.setdefault(c0, []).append((i, 0))
        adj.setdefault(c1, []).append((i, 1))

    used = bytearray(len(segs))
    lines = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = 1
        chain = [segs[i][0], segs[i][1]]
        cur = code_list[i][1]
        for direction in (1, 0):
            if direction == 0:
                chain.reverse()
                cur = code_list[i][0]
            while True:
                nxt = None
                for j, side in adj.get(cur, ()):
                    if not used[j]:
                        nxt = (j, side)
                        break
                if nxt is None:
                    break
                j, side = nxt
                used[j] = 1
                chain.append(segs[j][1 - side])
                cur = code_list[j][1 - side]
        lines.append(np.array(chain))
    return lines


def meta_radius(R, threshold):
    """Radius of a LONE ball's isosurface -- solved from the field, not traced.

    f = (5/9)w^2 + (4/9)w^3 with w = 1 - (r/R)^2, monotone in w, so bisect.
    An independent yardstick: a tracer that returns R instead of this is
    intersecting the bounding sphere rather than the metaball.
    """
    lo, hi = 0.0, 1.0
    for _ in range(80):
        w = 0.5 * (lo + hi)
        if (5.0 / 9.0) * w * w + (4.0 / 9.0) * w ** 3 < threshold:
            lo = w
        else:
            hi = w
    w = 0.5 * (lo + hi)
    return R * math.sqrt(max(1.0 - w, 0.0))


def paper_control_points(a):
    """Control-point y values straight from eq. (10) of the paper."""
    return np.array([0.0,
                     0.0,
                     16.0 / 27.0 * a**2,
                     8.0 * a**2 * (8.0 * a + 5.0) / 45.0,
                     16.0 / 27.0 * a**2,
                     0.0,
                     0.0])


# ==========================================================================
# 5. Presets
# ==========================================================================

PRESETS = {
    "Two balls, offset": {
        "spheres": [[3.0, 0.9, 1.3], [3.0, -1.9, 2.0]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Two balls, small + large": {
        "spheres": [[2.0, 0.6, 1.0], [3.0, -1.5, 2.0]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Single ball": {
        "spheres": [[3.0, 0.0, 1.5]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Two disjoint": {
        "spheres": [[2.0, 0.0, 0.8], [5.0, 0.0, 0.8]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Partial overlap": {
        "spheres": [[2.2, 0.0, 1.2], [4.0, 0.0, 1.2]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Full containment": {
        "spheres": [[3.0, 0.0, 2.2], [3.0, 0.0, 0.9]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Concentric": {
        "spheres": [[3.0, 0.0, 2.0], [3.0, 0.0, 2.0]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Tangent": {
        "spheres": [[2.0, 0.0, 1.0], [4.0, 0.0, 1.0]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Grazing ray": {
        "spheres": [[3.0, 1.45, 1.5]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Origin inside ball": {
        "spheres": [[0.5, 0.0, 1.5], [3.5, 0.0, 1.2]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Three mutually overlapping": {
        "spheres": [[2.0, 0.0, 1.3], [3.2, 0.0, 1.3], [4.4, 0.0, 1.3]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Overlap, reversed input order": {
        "spheres": [[4.0, 0.0, 1.2], [2.2, 0.0, 1.2]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    # Blending demos. In each of these the dashed lone-ball surfaces are
    # separate but the green isosurface is one merged shape -- that gap is the
    # blending, and it is what a sphere intersection cannot produce.
    "Blob: two merging": {
        "spheres": [[2.6, 0.62, 1.2], [2.6, -0.62, 1.2]],
        "origin": [-1.5, 0.0], "angle": 0.0,
    },
    "Blob: chain of four": {
        "spheres": [[2.0, 0.0, 1.0], [3.0, 0.55, 1.0],
                    [4.0, 0.0, 1.0], [5.0, 0.55, 1.0]],
        "origin": [-1.5, 0.0], "angle": 0.0,
    },
    "Blob: neck (barely merged)": {
        "spheres": [[3.0, 0.90, 1.2], [3.0, -0.90, 1.2]],
        "origin": [-1.5, 0.0], "angle": 0.0,
    },
    # The two cases below are the ones that actually exercise the compositing
    # path: neither ball alone reaches the threshold along this ray, so the
    # surface only exists because the two densities sum inside the overlap.
    "Grazing pair (blend-only hit)": {
        "spheres": [[2.25, 0.806, 1.3], [3.75, 0.806, 1.3]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
    "Grazing pair (no surface)": {
        "spheres": [[2.4, 1.15, 1.3], [3.6, 1.15, 1.3]],
        "origin": [0.0, 0.0], "angle": 0.0,
    },
}
