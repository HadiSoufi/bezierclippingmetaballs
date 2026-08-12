"""Line-by-line transliteration of Metaballs.hlsl into Python.

Checked against the brute-force solver in bezier_metaballs_core. Both shader
paths are measured -- the sweep and the single-sphere fast path -- plus
occlusion and the normal, none of which any other check covers.

Mirror every shader edit here and re-run before trusting it. Reasoning about
the root search has missed every real bug this project has had.

    py -3.10 harness.py --seed 4242
    py -3.10 harness.py --seed 31337 --rays 6000

NOTE: this harness is float64 and the shader is float32, so it CANNOT see
precision failures. Constants that exist to keep float32 out of trouble -- the
clip_len convergence break in particular -- must match the shader rather than
what float64 would tolerate.

NOT COVERED, and each of these has hidden a real bug:

  * The contact rim. The shader bends Normal into the view plane within
    RimThickness of the occluder and there is no ground truth for it -- the
    reference normal is the analytic field gradient, which is exactly what the
    bend departs from. Every ray here runs at scene_dist = 1e30, so the rim
    saturates to 0 and is inert; that is why the normal figures are unaffected.
  * The camera-inside normal. Both paths return without touching Normal, so it
    is whatever the initialiser left. Excluded from the normal statistic.
  * scene_dist. It is a plain parameter here; the shader derives it as
    dot(WorldPosition - TraceStart, TraceDirection), so a wrong TraceDirection
    pin is invisible to every test in this file.
  * The DATA. Nothing here can see Count, the Spheres layout, or whether the
    balls that reach the shader are the balls that exist. A clean run says the
    algorithm is right, not that the render is.

There is a second, stronger check that does not share this file's assumptions:
hlsl_compat.h + verify.cpp compile and RUN Metaballs.hlsl itself as C++ (only
[unroll] is stripped) against the same brute-force solver, in 3D. Use this file
to iterate, that one to confirm. They agree on 24000 identical rays to 4.2e-05,
which is the clip_len convergence bound.
"""
import math
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bezier_metaballs_core as C

DEBUG = False
K2 = 5.0 / 9.0
K3 = 4.0 / 9.0

NCLIP = 12          # clipping iterations; must match the loop bound in the shader
ACCEPT = 1e-4       # |density - Threshold| still allowed at the accepted root
MAXHIT = 32         # chords[] capacity in the shader; indexed by per-ray
                    # intersections, NOT by scene ball count


def _bernstein(dens):
    """Power basis -> degree-6 Bernstein control points.

    b_j = sum_k C(j,k)/C(6,k) * dens[k]. The ratio telescopes --
    C(j,k)/C(6,k) = prod_m (j-m)/(6-m) -- so each point is a nested Horner in
    a_k = (j-k+1)/(7-k), needing no table. At k = j+1 the factor is exactly 0,
    which discards the tail, so b_j reduces to its own j+1 terms.

    This mirrors the shader, which writes it as a pair of [unroll] loops and
    lets fxc fold the zeros: 3 instructions cheaper than the six control points
    written out by hand. The reassociation is not bit-identical to the explicit
    sum, but it was checked in float32 across four seeds with no difference in
    any hit distance.
    """
    out = [dens[0]]
    for j in range(1, 7):
        h = dens[6]
        for k in range(6, 0, -1):
            h = dens[k - 1] + h * ((j - k + 1) / float(7 - k))
        out.append(h)
    return tuple(out)


def _bernstein_explicit(dens):
    """The hand-expanded form, kept as a cross-check on _bernstein."""
    return (
        dens[0],
        dens[0] + dens[1] / 6.0,
        dens[0] + dens[1] / 3.0 + dens[2] / 15.0,
        dens[0] + dens[1] / 2.0 + dens[2] / 5.0 + dens[3] / 20.0,
        dens[0] + 2.0 * dens[1] / 3.0 + 2.0 * dens[2] / 5.0 + dens[3] / 5.0 + dens[4] / 15.0,
        dens[0] + 5.0 * dens[1] / 6.0 + 2.0 * dens[2] / 3.0 + dens[3] / 2.0
        + dens[4] / 3.0 + dens[5] / 6.0,
        dens[0] + dens[1] + dens[2] + dens[3] + dens[4] + dens[5] + dens[6],
    )


def _eval(dens, t):
    """Horner on the power basis."""
    g = dens[6]
    for i in (5, 4, 3, 2, 1):
        g = g * t + dens[i]
    return g * t + dens[0]


def _reparam(dens, a, h):
    """Restrict dens to [a, a+h], in place: G(sigma) = dens(a + h*sigma)."""
    for kfac in range(6):                     # Taylor shift by a, 21 mul-adds
        for i in range(5, kfac - 1, -1):
            dens[i] += a * dens[i + 1]
    p = 1.0                                 # scale by h
    for i in range(1, 7):
        p *= h
        dens[i] *= p


# NOTE: scene_dist is a plain parameter here. The shader derives it as
# dot(WorldPosition - TraceStart, TraceDirection) -- a projection, valid only
# because WorldPosition lies ON the ray. Nothing in this file exercises that
# derivation, so a bad TraceDirection pin would be invisible to these tests.
def trace(balls, TraceStart, TraceDirection, Threshold,
             scene_dist=1e30):
    D = np.asarray(TraceDirection, float)
    D = D / np.linalg.norm(D)
    O = np.asarray(TraceStart, float)

    # ---- loader (identical to v3) ----
    cen = []
    rng = []
    next_cut = 1e30
    for (cx, cy, R) in balls:
        center = np.array([cx, cy], float)
        fc = center - O                   # centre <- camera, as the shader does

        # Perpendicular-offset discriminant, NOT b*b - 4*c: at distance those
        # terms are both ~4d^2 and cancel to ~4R^2, which float32 cannot hold.
        # fc runs centre <- camera, so h is ALREADY the negated projection and
        # neither root needs a negate; flipping fc flips perp too, and
        # dot(perp, perp) does not care about the sign.
        h = np.dot(fc, D)
        perp = fc - h * D
        disc = R * R - np.dot(perp, perp)
        if disc < 0:                      # ray misses this sphere
            continue
        sq = math.sqrt(disc)
        t0 = h - sq
        t1 = h + sq
        if t1 <= 0:                       # sphere entirely behind us
            continue
        if t0 >= scene_dist:              # chord starts past the occluder: the
            continue                      # ball can never affect a visible hit
        if len(cen) >= MAXHIT:            # array full -- the shader's loop
            break                         # condition stops it here too
        cen.append(center)
        # The shader keeps ONLY this in registers -- plus the source texel
        # index -- and re-fetches the centre from Spheres in the tail, because
        # on RDNA3 the arrays are promoted to VGPRs and every element costs
        # occupancy. cen[] below stands in for that re-fetch; the values are
        # identical, so this still mirrors it.
        rng.append((t0, t1, 1.0 / (R * R)))
        if min(t0, t1) > 0.0:
            next_cut = min(next_cut, min(t0, t1))
        if max(t0, t1) > 0.0:
            next_cut = min(next_cut, max(t0, t1))

    n = len(cen)
    if n == 0:
        return None, None
    if n == 1:
        # Single-sphere fast path. Returns the OUTWARD normal, exactly what the
        # shader leaves in Normal -- this used to return None, so the path went
        # untested and shipped a normal that was up to 180 degrees wrong.
        center = cen[0]
        radius = (0.74602807 * 0.74602807) / rng[0][2]      # (0.746*R)^2
        fc = O - center
        c = np.dot(fc, fc) - radius
        if c <= 0:
            # Camera inside: the shader returns without touching Normal, which
            # is still TraceDirection from the top. Not a surface normal -- a
            # flat value the material is built around.
            return 0.0, D
        h = np.dot(fc, D)
        perp = fc - h * D
        disc = radius - np.dot(perp, perp)
        if h >= 0 or disc < 0:
            return None, None
        t = -h - math.sqrt(disc)
        if t >= scene_dist:               # behind the depth buffer
            return None, None
        P = O + t * D
        nn = np.linalg.norm(P - center)
        return t, ((P - center) / nn if nn > 0 else -D)

    seg_start = 0.0
    seg_end = 0.0
    surface = -1.0

    for _s in range(2 * n):
        if surface >= 0.0:
            break

        seg_start = seg_end
        seg_end = next_cut
        if seg_end > 1e29:
            break
        if seg_start >= scene_dist:           # past the depth buffer
            break

        seg_len = seg_end - seg_start
        seg_len_sq = seg_len * seg_len
        seg_mid = 0.5 * (seg_start + seg_end)

        # ---- composite (identical to v3) ----
        dens = [0.0] * 7
        next_cut = 1e30
        on = 0.0
        for k in range(n):
            if rng[k][0] > seg_end:
                next_cut = min(next_cut, rng[k][0])
            if rng[k][1] > seg_end:
                next_cut = min(next_cut, rng[k][1])

            # 57.7% of ball-visits do not cover the segment (measured over
            # 78623 visits). Skip rather than mask: with a mask det would be 0
            # and every term below would add exactly 0, so this is equivalent,
            # not an approximation.
            if rng[k][0] > seg_mid or rng[k][1] < seg_mid:
                continue
            on = 1.0                          # segment has at least one ball
            det = rng[k][2]
            p0 = seg_start - rng[k][0]
            p1 = seg_start - rng[k][1]

            # w(s) = w[0] + w[1]*s + w[2]*s^2
            w = (-(p0 * p1) * det,
                  -(seg_len * (p0 + p1)) * det,
                  -seg_len_sq * det)

            w2 = (w[0] * w[0],                        # w^2, degree 4
                 2.0 * w[0] * w[1],
                 w[1] * w[1] + 2.0 * w[0] * w[2],
                 2.0 * w[1] * w[2],
                 w[2] * w[2])

            kfac = (K2 + K3 * w[0],                     # K2 + K3*w, degree 2
                  K3 * w[1],
                  K3 * w[2])

            # f = K2*w^2 + K3*w^3 = w2 * kfac, a polynomial PRODUCT. Walking it as
            # dens[n+m] += w2[n]*kfac[m] is the same 15 terms as writing the seven
            # coefficient lines out; the tight bounds generate the triangle's
            # zeros structurally rather than computing them. The shader spells
            # this as two [unroll] loops.
            # NOT m/n -- `n` is the intersecting-ball count in this scope and
            # the normal loop below still needs it.
            for mi in range(3):
                for ni in range(5):
                    dens[ni + mi] += w2[ni] * kfac[mi]

        # 32.9% of segments are gaps with nothing covering them; their
        # polynomial is identically zero, so skip the whole clip loop.
        if on == 0.0:
            continue

        # ---- NEW: bracketed Bezier clipping ----
        # Invariant: [clip_start, clip_start + clip_len] contains the first root of the
        # segment, if the segment has one. Every bound below is a convex-hull
        # bound, so a root can never be stepped over.
        clip_start = 0.0
        clip_len = 1.0
        b0 = 0.0
        alive = True        # cleared only by a PROVEN no-root exit
        done = False        # converged; stop clipping

        for _it in range(NCLIP):
            if done:
                break
            bc = _bernstein(dens)
            b0 = bc[0]

            # Peak of the control polygon, and where it sits. The curve's own
            # maximum lies near it, which makes it far and away the best single
            # place to look for a point already over Threshold -- see the probe
            # below.
            # Peak, its position, and the extreme slopes out of b0, all folded
            # in ONE pass -- the shader does this too, so every test below has
            # them available before it runs.
            bern_max = bc[0]
            bern_at = 0.0
            slope_max = -1e30
            slope_min = 1e30
            for j, mul in ((1, 6.0), (2, 3.0), (3, 2.0),
                           (4, 1.5), (5, 1.2), (6, 1.0)):
                if bc[j] > bern_max:
                    bern_max = bc[j]
                    bern_at = j / 6.0
                sl = (bc[j] - b0) * mul
                slope_max = max(slope_max, sl)
                slope_min = min(slope_min, sl)

            if b0 >= Threshold:
                # At or past the surface at the interval start -- either
                # converged, or the segment opens inside the isosurface. Either
                # way clip_start is the answer.
                #
                # THIS MUST COME FIRST. Testing slope_max <= 0 ahead of it rejects a
                # segment that starts inside the isosurface with the density
                # falling outward, which has to hit at seg_start: measured 124 such
                # rays in 2373 and up to 9.7 world units of error.
                #
                # STRICT, unlike the hull test below. ACCEPT is hull slack only;
                # reusing it here would stop as soon as the density came within
                # 1e-4 of Threshold, which is ~4e-7 of position error on a
                # typical ray -- eight orders worse than clipping can do.
                done = True
                break

            # Convex hull property: all control points below Threshold means the
            # curve is too, so there is no crossing left in here. Or the polygon
            # never rises out of b0, so it cannot climb to Threshold either. The
            # shader merges these into one branch, which is safe only under the
            # accept above -- bern_max >= b0, so the hull half can never fire on
            # a segment the accept would have taken.
            #
            # The ACCEPT slack is load-bearing, not sloppiness. As clip_len
            # collapses onto the root the reparameterised curve tends to the
            # CONSTANT dens(clip_start), which sits just below Threshold -- so a strict
            # test rejects an answer that has already converged, and running MORE
            # iterations makes the result worse. Same shape as the v3 bug where a
            # strict interval test discarded a converged Newton iterate.
            if bern_max < Threshold - ACCEPT or slope_max <= 0.0:
                alive = False
                break

            g0 = b0 - Threshold         # < 0 here
            lo = -g0 / slope_max            # curve is provably below Threshold here
            if lo >= 1.0:
                alive = False
                break

            # Upper bound: zero of the shallowest line out of b0.
            hi = (-g0 / slope_min) if slope_min > 1e-12 else 1e30

            if hi >= 1.0:
                # The shallowest line does not reach Threshold inside this
                # interval -- which is the NORMAL case whenever the span ends
                # below Threshold, i.e. any segment where the density peaks and
                # comes back down. Without a real upper bound this degrades to
                # left-only clipping and converges linearly (~30 iterations).
                # bezier_metaballs_core._probe_bracket exists for exactly this.
                #
                # The scan starts at `lo`, the PROVEN hull bound, not at the
                # previous probe. That is the whole correctness fix over v3:
                # v3 narrowed the left end to the previous probe, so a density
                # bump rising and falling between two probes put the first root
                # outside the bracket and Newton converged on a later one.
                # Here the interval always still contains the first root, so a
                # missed bump only costs convergence rate, never correctness.
                hi = 1.0

                # The control-polygon peak, and ONLY the peak. A uniform grid can
                # step straight over a thin bump: on a grazing hit the density
                # can clear Threshold by ~5e-4 across a window a fiftieth of the
                # segment wide, and even probes simply straddle it. The peak is
                # where the curve is highest, so it is the one place worth a
                # Horner evaluation.
                #
                # A 4-probe fallback grid used to follow this. It ran on 2% of
                # iterations, found anything on 0.1%, and removing it was
                # byte-identical here across three seeds -- same p50, max, false,
                # miss and normal error -- while taking the shader's clip loop
                # from 163 instructions to 134.
                if bern_at > lo and _eval(dens, bern_at) >= Threshold:
                    hi = bern_at

            if hi <= lo:
                hi = 1.0

            clip_start += clip_len * lo
            clip_len *= (hi - lo)

            # Stop before the interval degenerates. 1e-5, NOT 1e-12: the
            # shader is float32 and the reparameterised F6 carries clip_len^6,
            # which underflows past 1.2e-38 long before 1e-12. Beyond that the
            # slopes are noise and the "no root" exits fire at random -- the
            # pinhole rings. This harness is float64 and cannot see it, so the
            # constant must match the shader rather than what float64 allows.
            if clip_len <= 1e-5:
                done = True
                break

            _reparam(dens, lo, hi - lo)

        # Accept whenever nothing PROVED there is no root. clip_start is a proven
        # lower bound on the first root that converges up onto it, so even an
        # unconverged clip_start is inside the right segment and off by at most
        # clip_len. Rejecting instead would send the sweep on to a LATER segment
        # and answer a completely different surface -- far worse than a slightly
        # early root here.
        if alive:
            surface = seg_start + clip_start * seg_len

    if surface < 0.0:
        return None, None
    if surface >= scene_dist:             # crossing landed behind the occluder
        return None, None

    if surface <= 0.0:
        # Camera inside: faces the camera, matching the fast path. Returns
        # rather than falling through so nothing downstream rotates it -- in the
        # shader the contact rim would otherwise bend it into the view plane
        # across the entire screen, since once inside EVERY ray reports inside.
        return 0.0, D

    P = O + surface * D
    grad = np.zeros(len(O))
    for k in range(n):
        dv = P - cen[k]
        det = rng[k][2]
        u = np.dot(dv, dv) * det
        if u >= 1.0:
            continue
        grad = grad + ((-12.0 / 9.0) * u * u + (34.0 / 9.0) * u
                       - (22.0 / 9.0)) * 2.0 * det * dv
    # OUTWARD, as the shader returns it: the field falls off outward so the
    # gradient points inward, and the shader negates it.
    nn = np.linalg.norm(grad)
    return surface, (-grad / nn if nn > 0 else D)   # degenerate: same painted
                                                    # constant as the inside paths


def _surviving(balls, O, D):
    O = np.asarray(O, float); D = np.asarray(D, float); D = D / np.linalg.norm(D)
    n = 0
    for (cx, cy, R) in balls:
        fc = O - np.array([cx, cy])
        h = np.dot(fc, D)
        perp = fc - h * D
        disc = R * R - np.dot(perp, perp)
        if disc < 0 or -h + math.sqrt(disc) <= 0:
            continue
        n += 1
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.148,
                    help="the shader hardcodes THRESHOLD = 0.148 as a static "
                         "float; any other value here is a what-if, not what "
                         "ships")
    ap.add_argument("--rays", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--clip", type=int, default=None,
                    help="override NCLIP for a convergence sweep")
    a = ap.parse_args()

    global NCLIP
    if a.clip:
        NCLIP = a.clip

    rng = np.random.default_rng(a.seed)
    # BOTH code paths are measured. The fast path used to be skipped entirely,
    # which is how it shipped a normal up to 180 degrees out.
    st = {1: [[], 0, 0, 0, []], 2: [[], 0, 0, 0, []]}   # err, false, miss, tot, nerr
    for _ in range(a.rays):
        k = int(rng.integers(1, 9))
        sph = [[float(rng.uniform(1.0, 5.0)), float(rng.uniform(-1.5, 1.5)),
                float(rng.uniform(0.4, 1.7))] for _ in range(k)]
        inside = rng.random() < 0.3
        o = (np.array([float(rng.uniform(1.0, 5.0)),
                       float(rng.uniform(-1.5, 1.5))])
             if inside else np.array([-1.0, float(rng.uniform(-2.5, 2.5))]))
        th = float(rng.uniform(-0.6, 0.6))
        d = [math.cos(th), math.sin(th)]

        surv = _surviving(sph, o, d)
        if surv == 0:
            continue
        w2 = st[1 if surv == 1 else 2]
        w2[3] += 1
        ref, refp, refn = C.reference_hit(sph, o, d, a.threshold)
        got, gotn = trace(sph, o, d, a.threshold)
        if ref is None and got is not None:
            w2[1] += 1
        elif ref is not None and got is None:
            w2[2] += 1
        elif ref is not None:
            w2[0].append(abs(got - ref))
            # Camera-inside rays are excluded from the NORMAL statistic only:
            # the shader deliberately returns -D there rather than the field
            # gradient, so refn (which IS the gradient) is not ground truth for
            # them. Their POSITIONS are still checked, above.
            if (gotn is not None and np.linalg.norm(refn) > 0
                    and got > 0.0):
                w2[4].append(float(np.degrees(np.arccos(
                    np.clip(np.dot(gotn, refn), -1, 1)))))

    # ---- occlusion invariants -------------------------------------------
    # Never exercised before this session: scene_dist was 1e30 in every run,
    # which is exactly why a bad occlusion input went unnoticed for so long.
    occ_bad = occ_tot = 0
    rng2 = np.random.default_rng(a.seed ^ 0x5EED)
    for _ in range(600):
        k = int(rng2.integers(2, 7))
        sph2 = [[float(rng2.uniform(1.0, 5.0)), float(rng2.uniform(-1.5, 1.5)),
                 float(rng2.uniform(0.4, 1.7))] for _ in range(k)]
        o2 = np.array([-1.0, float(rng2.uniform(-2.0, 2.0))])
        th2 = float(rng2.uniform(-0.5, 0.5))
        d2 = [math.cos(th2), math.sin(th2)]
        base, _ = trace(sph2, o2, d2, a.threshold)
        if base is None or base <= 0:
            continue
        occ_tot += 1
        # occluder just behind the hit -> must still be drawn
        keep, _ = trace(sph2, o2, d2, a.threshold, scene_dist=base * 1.001 + 1e-6)
        # occluder just in front -> must be culled
        cull, _ = trace(sph2, o2, d2, a.threshold, scene_dist=base * 0.999)
        if keep is None or cull is not None:
            occ_bad += 1

    print(f"Metaballs.hlsl vs brute force   Threshold={a.threshold}  "
          f"seed={a.seed}  NCLIP={NCLIP}")
    print("-" * 78)
    ok = True
    for key, name in ((2, "clipping (>=2 balls)"), (1, "fast path (1 ball)")):
        errs, f, m, tot, ne = st[key]
        e = np.array(errs) if errs else np.array([np.nan])
        n = np.array(ne) if ne else np.array([np.nan])
        print(f"  {name:22s} rays={tot:5d} p50={np.nanmedian(e):8.1e} "
              f"max={np.nanmax(e):8.1e} false={f:3d} miss={m:3d} "
              f"normal={np.nanmax(n):8.1e} deg")
        if f or m or not (np.nanmax(e) < 1e-3) or not (np.nanmax(n) < 1e-1):
            ok = False
    print(f"  {'occlusion':22s} rays={occ_tot:5d} "
          f"violations={occ_bad:3d}  (occluder just behind must draw, "
          f"just in front must cull)")
    if occ_bad:
        ok = False
    print("-" * 78)
    print("  harness:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
