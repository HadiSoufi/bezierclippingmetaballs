// ===========================================================================
// Bezier-clipped metaballs -- Unreal Custom node body.
//
// The ray is cut at every chord end; between cuts the covering set is constant,
// so each ball's density is a degree-6 polynomial in t and overlapping balls
// composite by adding coefficients. A convex-hull test rejects segments that
// cannot hold the surface; bracketed Bezier clipping finds the first root in
// the rest. This is bezier_metaballs_core.first_root with two shader changes:
// the upper bracket is the shallowest control-point slope rather than a
// monotone-chain hull (no data-dependent loop), and clipping reparameterises
// the power basis by Taylor shift + scale rather than de Casteljau (no array).
//
// Every bound is a convex-hull bound, so the first root cannot be stepped over.
// A segment is accepted unless something PROVES it root-free: clip_start is a lower
// bound rising onto the root, so an unconverged answer is early within the
// right segment, where rejecting would draw a later surface entirely.
//
// Comments marked DO NOT each cost a debugging pass. Mirror any change to the
// root search into harness.py and run it before trusting the change.
// ===========================================================================
if (Count == 0) return 0;

Normal = TraceDirection;
Position = TraceStart;
// Distance to the occluder along THIS ray. A projection, not a length:
// WorldPosition is the scene point at this pixel and TraceDirection is
// normalize(WorldPosition - CameraPosition), so the point lies ON the ray and
// the dot IS the distance -- no sqrt, and signed, which is what every
// comparison below actually wants. Measured 5 instructions cheaper than
// length(). If TraceDirection is ever fed from something other than that
// difference (CameraVector, say), this stops being equivalent.
const float NEAREST_SURFACE = dot(WorldPosition - TraceStart, TraceDirection);

// The Spheres render target is 256x256 and that is HARDCODED here. It buys a
// lot: no GetDimensions (a resinfo every pixel for a frame constant), and
// because 256 is a power of two the row-major walk becomes pure bit math off
// the loop counter -- no cursor, no wrap test, no conditional moves. Measured
// 6 instructions off the loader body, which runs Count times per pixel.
//
// IF THE RT IS EVER RESIZED, CHANGE BOTH CONSTANTS, and keep it a power of two.
// Nothing checks this: a mismatch silently reads the wrong texels, so balls
// land at wrong positions rather than failing loudly. Max Count is 256*256.
static const uint RT_MASK  = 255;   // width - 1
static const uint RT_SHIFT = 8;     // log2(width)

// Split by consumer so each loop gets what it needs from ONE float4 load: the
// sweep reads chords, the normal and fast paths read balls. 1/R^2 is in both
// because both need it and the component exists anyway. R is the SCALED radius,
// which is what keeps the fast path's 0.746 consistent with the sweep.
// ONE array, not two. On RDNA3 both arrays are promoted into registers rather
// than scratch (measured: 0 bytes scratch, 233 VGPRs), so every element costs
// register file and therefore occupancy directly. balls[32] as a float4 was 128
// of those VGPRs while being read only TWICE per pixel -- balls[0] on the fast
// path and once per ball in the gradient tail, neither in the hot sweep.
//
// Storing the source texel index instead and re-fetching the centre from
// Spheres costs a cached, wave-coherent load in a once-per-pixel loop and buys
// 233 -> 175 VGPRs, 6 -> 8 waves/SIMD, with the 32 cap untouched. It also drops
// one of the two rcp rounds on the lone-ball radius.
//
// Measured with RGA (gfx1100). The cap is the other occupancy lever -- 16 would
// give 10 waves -- but dense scenes already exceed 32 intersections per ray, so
// shrinking it trades silent, texture-ordered ball drops for the waves.
float4 chords[32];          // x = start, y = end, z = 1/R^2, w = texel index

// SCRATCH REGISTERS, reused across stages on purpose -- disjoint lifetimes cost
// nothing, and naming them for one stage would misname them for the others.
// What each holds, in order of appearance:
//
//   fc     loader: centre <- camera, then the perpendicular offset
//          fast path: camera <- centre, then the perpendicular offset
//          tail: the summed field gradient
//   entry  loader: the raw Spheres fetch
//          composite: chords[j]   |   tail: balls[i], then centre -> Position
//   det    loader / fast path: ray parameter at closest approach
//          composite: 1/R^2 for the ball being added
//   b      loader: discriminant, then tmin
//          clip loop: dens[0] - THRESHOLD, the deficit the brackets divide
//          composite: next-cut candidate, then seg_start - tmin
//          fast path: discriminant, then the rim blend factor
//   c      loader: half-chord, then tmax
//          composite: seg_start - tmax
//          fast path: camera-inside test, then half-chord, then |bent normal|^2
//   t0,t1  fast path: the hit distance
//          clip loop: the bracket [t0, t1], then the span and its running
//                     power once the bracket has been consumed
//          tail: u = (r/R)^2 for the ball being accumulated
float3 fc;
float4 entry;
float b, c, det, t0, t1;
int intersections = 0;
const int COUNT = (int) Count;
float next_cut = 1e30;      // first chord end ahead of the camera

int i;

// Load sphere data from rendertarget into working memory.
// The capacity test lives in the loop CONDITION, not as a break before the
// store: once the array is full there is nothing to gain from evaluating the
// next ball's geometry and then bailing. Two instructions off the body of the
// loop that runs Count times per pixel.
for (i = 0; i < COUNT && intersections < 32; i++) {
    entry = Spheres.Load(int3(i & RT_MASK, i >> RT_SHIFT, 0));

    if (!(entry.w > 0)) continue;

    fc = entry.xyz - TraceStart;                // centre <- camera; see below

    // Ray/sphere via the PERPENDICULAR OFFSET, not b*b - 4*c. DO NOT put that
    // back: at range both terms are ~4d^2 and differ by only ~4R^2, so float32
    // keeps none of it -- exactly 0.0 at d = 1e6, and at d = 1e5 it corrupted
    // 390 of 551 chords and invented 28 balls the ray never touched. Here the
    // projection comes off first, so nothing large ever cancels.
    // fc runs centre <- camera, so this dot is ALREADY the negated projection
    // and nothing below needs a negate. Flipping fc flips perp too, and
    // dot(perp, perp) does not care about its sign.
    det = dot(fc, TraceDirection);              // det IS the closest approach
    fc -= det * TraceDirection;                 // reuse as perp (negated, same length)
    b = entry.w - dot(fc, fc);                  // discriminant, monic form
    if (b < 0) continue;                        // ray misses this sphere

    // Roots are det -+ sqrt, already ordered -- no divide, no sign(), so the
    // old form's 0/0 NaN at b == 0 cannot happen.
    c = sqrt(b);
    b = det - c;
    c += det;
    if (c <= 0) continue;                       // sphere entirely behind us

    // Chord starts at or past the occluder, so this ball can never matter: any
    // accepted surface is < NEAREST_SURFACE <= tmin, the field is zero before
    // tmin, and at that Position the ball is outside its own radius so the
    // gradient skips it too. Dropping it shrinks intersections, which shrinks
    // the composite quadratically -- and usually to zero, taking the whole
    // sweep with it.
    //
    // MUST stay below the three continues above: it needs tmin, and placed
    // here it runs only for balls that already intersect the ray ahead of the
    // camera -- ~1.3 per ray, not Count. Measured 2.6 instructions against
    // 139-255 saved. The exception is an occluder beyond every ball, where it
    // removes nothing and costs the 2.6.
    if (b >= NEAREST_SURFACE) continue;         // chord starts past the occluder

    // Entry stored RAW, not clamped to 0 when the camera is inside: seg_start and
    // seg_mid are always >= 0 so the sweep cannot tell, and the field needs the true
    // root. entry is dead here, so .w carries the reciprocal into both stores.
    entry.w = rcp(entry.w);
    chords[intersections]  = float4(b, c, entry.w, (float) i);
    intersections++;

    // First breakpoint ahead of the camera. Only ONE candidate needs testing:
    // tmax > 0 is proven by the `continue` above and tmin <= tmax, so it is
    // tmin when tmin is ahead, tmax otherwise.
    next_cut = min(next_cut, (b > 0.0) ? b : c);
}

if (intersections == 0) return 0;

// One ball: solve analytically. 0.74602807 solves (5/9)w^2 + (4/9)w^3 for
// THRESHOLD 0.148. Both paths must agree on the lone-ball radius or the surface
// pops the moment a second ball comes into range.
if (intersections == 1) {
    i = (int) chords[0].w;                      // re-fetch ball 0
    entry = Spheres.Load(int3(i & RT_MASK, i >> RT_SHIFT, 0));
    entry.w *= 0.5565578812;                    // (0.746*R)^2, one rcp fewer

    fc = TraceStart - entry.xyz;
    c = dot(fc, fc) - entry.w;       // <= 0 means the camera is inside

    // Camera inside the isosurface: stop. Normal is still TraceDirection from
    // the top -- nothing writes it on the way here -- and that IS what the graph
    // wants inside, so this returns without spending an instruction restating it.
    if (c <= 0) return 1;

    // Same perpendicular-offset form as the loader, same reason. fc is dead, so
    // it becomes the perpendicular vector; the centre is not needed again.
    det = dot(fc, TraceDirection);
    fc -= det * TraceDirection;
    b = entry.w - dot(fc, fc);                   // discriminant
    if (det >= 0 || b < 0) return 0;            // pointing away, or misses

    c = sqrt(b);                                // half-chord of the meta sphere
    t0 = -det - c;                              // distance along the ray
    if (t0 >= NEAREST_SURFACE) return 0;             // behind the depth buffer
    Position = t0 * TraceDirection + TraceStart;

    // Normal without subtracting two world positions. Exactly:
    //   Position - center = (perp + det*D) + (-det - c)*D = perp - c*D
    // and fc already holds perp. Exact algebra, not an approximation.
    //
    // DO NOT go back to normalize(Position - center): both are absolute world
    // coordinates differing by only ~0.746*R, so that form discards every digit
    // they share -- and the level's world origin, not the ball, decides how many
    // that is. |perp|^2 + c^2 == radius2 by construction, so rsqrt of the stored
    // radius2 replaces the normalize too.
    Normal = (fc - c * TraceDirection) * rsqrt(entry.w);

    // Contact rim, world units. b, c and fc are dead here; the guard covers a
    // normal exactly along the ray, where the bent vector has zero length.
    b = saturate(1.0 - (NEAREST_SURFACE - t0) / max(RimThickness, 1e-6));
    fc = Normal - b * dot(Normal, TraceDirection) * TraceDirection;
    c = dot(fc, fc);
    Normal = (c > 1e-8) ? fc * rsqrt(c) : Normal;
    return 1;
}

// With tmin/tmax the chord ends and u = (r/R)^2,
//     w(t) = 1 - u = -(t - tmin)(t - tmax) / R^2
// and f = (5/9)w^2 + (4/9)w^3, degree 6 in t. w is zero at the chord ends by
// construction. The root form avoids the cancellation t^2 + b*t + c suffers at
// world scale.
const float K2 = 5.0 / 9.0;
const float K3 = 4.0 / 9.0;
const float THRESHOLD = 0.148;
const float ACCEPT = 1e-4;  // hull-rejection slack; see the hull test

// Composite density over the segment, power basis: f = sum dens[n]*s^n.
float dens[7];              // composite density, power basis: sum dens[n]*s^n
float w[3];                 // w(s) = w[0] + w[1]*s + w[2]*s^2
float w2[5];                // w*w, degree 4
float kfac[3];              // K2 + K3*w

float seg_start = 0.0;      // segment start, world distance along the ray
float seg_end = 0.0;        // segment end
float seg_len = 0.0;        // seg_end - seg_start
float seg_len_sq = 0.0;
float seg_mid = 0.0;
float bern = 0.0;           // a Bernstein control point, then its slope out of b0
float bern_max = 0.0;       // largest control point -> the safety test
float bern_at = 0.0;        // where it sits, 0..1 -> the first probe
float slope_max = 0.0;      // steepest rise out of b0 -> lower bracket
float slope_min = 0.0;      // shallowest rise out of b0 -> upper bracket
float clip_start = 0.0;     // clipped interval start, 0..1 within the segment
float clip_len = 1.0;       // width of that interval, 0..1
float surface = -1.0;       // hit distance; < 0 means none found yet

int j, k, m;                // iterators: j outer, k/m nested inside it

bool alive = false;         // true while nothing has PROVED the segment root-free
bool covered = false;       // true if any ball covers the current segment

// Sweep front to back. At most 2*intersections chord ends bounds the loop.
for (i = 0; i < 2 * intersections && surface < 0.0; i++) {

    // Advance here rather than at every `continue`. Both start at 0 and next_cut
    // holds the first breakpoint, so pass one is [0, first].
    seg_start = seg_end;
    seg_end = next_cut;

    // ray has left every ball or past the depth buffer
    if (seg_end > 1e29 || seg_start >= NEAREST_SURFACE) break;

    seg_len = seg_end - seg_start;
    seg_len_sq = seg_len * seg_len;
    seg_mid = 0.5 * (seg_start + seg_end);

    // Composite the covering balls and find the next breakpoint in ONE pass.
    // Overlapping balls add coefficients; nothing needs clipping, because the
    // segment ends are chord ends so every active ball is fully on across it.
    [unroll]
    for (k = 0; k < 7; k++) dens[k] = 0;      // 7, not 6 -- dens[6] is a coefficient
    next_cut = 1e30;
    covered = false;

    for (j = 0; j < intersections; j++) {

        entry = chords[j];

        // Next breakpoint. tmin <= tmax, so the earlier end still ahead of
        // seg_end is tmin if tmin clears it, else tmax -- and if tmax does not
        // clear it, neither does tmin. One select and one min, not two of each.
        b = (entry.x > seg_end) ? entry.x : entry.y;
        next_cut = (entry.y > seg_end) ? min(next_cut, b) : next_cut;

        // Most visits do not cover. Skipping beats masking; the worst case costs
        // the compares a mask costs anyway. A bet on wave coherence.
        if (entry.x > seg_mid || entry.y < seg_mid) continue;

        covered = true;
        det = entry.z;                  // 1/R^2

        // t = seg_start + seg_len*s into -(t-tmin)(t-tmax)/R^2; b, c are the
        // shifted roots.
        b = seg_start - entry.x;
        c = seg_start - entry.y;

        // w2 = w^2 (degree 4) and kfac = K2 + K3*w (degree 2), so the density
        // f = K2*w^2 + K3*w^3 is the polynomial PRODUCT w2*kfac. dens[n+m] += w2[n]*kfac[m]
        // walks that product: 15 terms either way, and the tight bounds produce
        // the triangle's zeros structurally rather than computing them.
        w[0] = -(b * c) * det;
        w[1] = -(seg_len * (b + c)) * det;
        w[2] = -seg_len_sq * det;

        // w2 is w convolved with itself, and w is the convolution of its own two
        // linear factors, so both COULD be loops like the dens update below.
        // DO NOT: measured +3 instructions for w2, because the explicit form
        // exploits a symmetry the loop cannot -- 2*w[0]*w[1] is one product and
        // a scale, while the convolution computes w[0]*w[1] and w[1]*w[0]
        // separately and the compiler does not fold them.
        w2[0] = w[0] * w[0];
        w2[1] = 2.0 * w[0] * w[1];
        w2[2] = w[1] * w[1] + 2.0 * w[0] * w[2];
        w2[3] = 2.0 * w[1] * w[2];
        w2[4] = w[2] * w[2];

        kfac[0] = K2 + K3 * w[0];
        kfac[1] = K3 * w[1];
        kfac[2] = K3 * w[2];

        [unroll]
        for (k = 0; k < 3; k++)
            [unroll]
            for (m = 0; m < 5; m++)
                dens[m + k] += w2[m] * kfac[k];
    }

    // A GAP between balls: nothing covers this span, so every coefficient is
    // still zero and there is no surface to find. Purely an early-out
    // -- the clip loop rejects the flat zero correctly on its first pass, just
    // 49 instructions later. Break-even is ~0.24 empty segments per ray.
    if (!covered) continue;

    // Invariant: [clip_start, clip_start + clip_len] holds the segment's FIRST root if
    // it has one.
    clip_start = 0.0;
    clip_len = 1.0;
    alive = true;

    // DO NOT [unroll] -- every exit is data-dependent, so it buys no scheduling
    // and measured 2403 instruction slots against 484 rolled.
    for (j = 0; j < 12; j++) {

        // Bernstein control points b_k = sum_m C(k,m)/C(6,m) * dens[m], folded in
        // one pass with the polygon peak, its position, and the extreme slopes
        // out of b0 (slope_k = (b_k - b0)/(k/6), hence the 6/k).
        //
        // The ratio telescopes -- C(k,m)/C(6,m) = prod (k-i)/(6-i) -- so each
        // point is a nested Horner in a_m = (k-m+1)/(7-m), needing no table. At
        // m = k+1 the factor is exactly 0 and discards the tail, so after
        // [unroll] each b_k folds back to its own k+1 terms: 3 instructions
        // CHEAPER than writing the six control points out by hand.
        bern_max = dens[0];
        bern_at = 0.0;
        slope_max = -1e30;
        slope_min = 1e30;

        [unroll]
        for (k = 1; k <= 6; k++) {
            bern = dens[6];
            [unroll]
            for (m = 6; m >= 1; m--)
                bern = dens[m - 1] + bern * ((k - m + 1) / (float) (7 - m));

            bern_at  = (bern > bern_max) ? (k / 6.0) : bern_at;
            bern_max = max(bern_max, bern);
            bern = (bern - dens[0]) * (6.0 / k);   // bern is spent; reuse as the slope
            slope_max = max(slope_max, bern);
            slope_min = min(slope_min, bern);
        }

        // At or past the surface -- ACCEPT. DO NOT let this sink below the
        // reject: slope_max <= 0 with dens[0] >= THRESHOLD is a segment that STARTS
        // inside the isosurface and falls away, which must hit at seg_start rather
        // than be swept past -- 124 such rays in 2373, up to 9.7 world units of
        // error, with the order reversed. DO NOT reuse ACCEPT here either: it is
        // hull slack only, and stopping 1e-4 of density early is ~4e-7 of
        // position.
        if (dens[0] >= THRESHOLD) break;

        // Proven no root here, or the polygon never rises out of b0. These merge
        // safely ONLY under the accept above: bern_max >= dens[0], so the hull half
        // cannot fire on a segment the accept would have taken. DO NOT tighten to
        // `< THRESHOLD`: as clip_len collapses the curve tends to the CONSTANT
        // dens(clip_start) just below THRESHOLD, so a strict test throws away a
        // converged answer, and more iterations then make it worse.
        if (bern_max < THRESHOLD - ACCEPT || slope_max <= 0.0) { alive = false; break; }

        b = dens[0] - THRESHOLD;               // deficit; b is dead here                 // < 0 from here
        t0 = -b / slope_max;                    // provably below THRESHOLD before t0
        if (t0 >= 1.0) { alive = false; break; }

        // Upper bracket from the shallowest line. It lands outside the interval
        // whenever the span ends below THRESHOLD -- any segment where the
        // density peaks and comes back down -- so fall back to probing from t0,
        // the PROVEN bound. Probing from the previous probe instead is exactly
        // what let the older Newton search skip a root.
        t1 = (slope_min > 1e-12) ? (-b / slope_min) : 1e30;

        // The polygon peak, and ONLY the peak. DO NOT add a uniform probe grid:
        // a grazing hit can clear THRESHOLD by ~5e-4 across a fiftieth of a
        // segment and an even grid straddles it, while the peak is where the
        // curve is highest. A 4-probe grid used to sit here -- it ran on 2% of
        // iterations, found anything on 0.1%, and removing it was byte-identical
        // on the harness across three seeds.
        //
        // Finding no bound is still correct: t1 stays 1, the clip is left-ended
        // only, and clip_start still rises onto the root.
        if (t1 >= 1.0) {
            bern = dens[6];
            [unroll]
            for (m = 5; m >= 0; m--) bern = bern * bern_at + dens[m];

            t1 = (bern_at > t0 && bern >= THRESHOLD) ? bern_at : 1.0;
        }

        t1 = (t1 <= t0) ? 1.0 : t1;
        clip_start += clip_len * t0;
        clip_len *= (t1 - t0);

        // Converged. DO NOT tighten toward 1e-12 -- that is a float64 number and
        // the shader is float32. Reparameterised dens[6] carries clip_len^6: at 1e-5
        // that is ~1e-30 (fine), at 1e-12 it is 1e-72, far under float32's 1.2e-38
        // denormal floor. The high coefficients collapse, the slopes become noise,
        // and slope_max / t0 -- two near-zero quantities -- trips the "no root" exits
        // at random. Those are the pinhole rings: 0.17% of rays lost at 1e-12,
        // none at 1e-5, float32 only. Cost here is bounded by ~1e-5 of a segment.
        if (clip_len <= 1e-5) break;

        // Restrict dens to [t0, t1]: Taylor shift by t0, then scale by (t1 - t0).
        // 4x cheaper than de Casteljau, and needs no second array.
        [unroll]
        for (k = 0; k < 6; k++)
            [unroll]
            for (m = 5; m >= k; m--)
                dens[m] += t0 * dens[m + 1];

        // t0 and t1 are both finished with the bracket here, so they carry the
        // span and its running power instead of a fourth register.
        t1 -= t0;                   // the span
        t0  = t1;                   // span^1, then span^2, ...
        [unroll]
        for (k = 1; k < 7; k++) {
            dens[k] *= t0;
            t0 *= t1;
        }
    }

    surface = alive ? seg_start + clip_start * seg_len : surface;
}

if (surface < 0.0 || surface >= NEAREST_SURFACE) return 0;
if (surface <= 0.0) return 1;

Position = TraceStart + surface * TraceDirection;

// Normal: analytic gradient of the summed field. With u = (r/R)^2,
// df/du = -12/9 u^2 + 34/9 u - 22/9, negative for u < 1, so the field falls off
// outward: the gradient points INWARD and is negated for lighting.
fc = float3(0, 0, 0);
for (i = 0; i < intersections; i++) {
    j = (int) chords[i].w;                      // re-fetch the centre
    entry = Spheres.Load(int3(j & RT_MASK, j >> RT_SHIFT, 0));
    entry.w = chords[i].z;                      // 1/R^2 from the loader
    entry.xyz = Position - entry.xyz;
    t0 = dot(entry.xyz, entry.xyz) * entry.w;       // u
    if (t0 >= 1.0) continue;                        // outside this ball
    fc += (t0 * (t0 * -12.0 + 34.0) - 22.0) * (2.0 / 9.0) * entry.w * entry.xyz;
}

Normal = (dot(fc, fc) > 0.0) ? normalize(-fc) : TraceDirection;

// Contact rim, world units -- identical to the fast path, so the two agree as a
// ball comes into range. See the header.
b = saturate(1.0 - (NEAREST_SURFACE - surface) / max(RimThickness, 1e-6));
fc = Normal - b * dot(Normal, TraceDirection) * TraceDirection;
c = dot(fc, fc);
Normal = (c > 1e-8) ? fc * rsqrt(c) : Normal;
return 1;
