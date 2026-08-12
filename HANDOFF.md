# Handoff — Bézier-clipped metaballs

Read this first if you are picking the work up cold.

## The goal

Ray-trace metaball isosurfaces inside an **Unreal material Custom node**, replacing
a working-but-slow raymarched implementation. The method is Nishita/Nakamae Bézier
clipping over Wyvill's degree-6 field (`Bezier Clipping- Excerpt of Nishita et al,
1990.pdf` in this folder).

**Raymarching is off the table.** A raymarched version already exists and works;
the entire point is to beat it. Do not propose fixed-step sampling, sphere tracing,
or "scan then bisect" — that is the problem, not the solution.

## Constraints that shape everything

Unreal Custom nodes **cannot declare functions**, so nothing can be factored out —
every helper is written inline. Loops and branches are expensive, and the house
style deliberately reuses variables rather than declaring new ones. Match that.

The material is a **post-process**. That matters more than it sounds — see
Occlusion below.

**`THRESHOLD` is 0.148**, a `const float` in the shader. Not a pin. The lone-ball
surface radius matching it is `0.74602807 · R`, used by the single-sphere path and
stored pre-squared as `0.5565578812`. The two are solved from each other and must
change together.

## Files

| File | What it is |
|---|---|
| `Metaballs.hlsl` | **The deliverable.** The Custom node body. |
| `harness.py` | **Run this before trusting any shader change.** Line-by-line transliteration of `Metaballs.hlsl`, checked against the brute-force solver. Covers the sweep, the fast path, occlusion and normals. |
| *(scratchpad)* `hlsl_compat.h`, `verify.cpp` | Compiles and RUNS the real `Metaballs.hlsl` as C++ in 3D against brute force. Cannot drift from the shader, because it executes the shader. See Testing. |
| `bezier_metaballs_core.py` | Python reference: the corrected algorithm, a brute-force ground-truth field solver, presets. The oracle `harness.py` checks against. |
| `BezierMetaballsLab.py` | Interactive 2D lab — drop circles, drag a camera, watch the density curve and each clip iteration. |
| `Bezier Clipping- Excerpt of Nishita et al, 1990.pdf` | The paper. |

`harness.py` was `validate_v4.py`. The Newton-root-search variant and its harness
are gone for good.

## Custom node pins

```
Spheres         Texture Object    .rgb = centre, .a = radius SQUARED and SCALED
                                  **256x256, HARDCODED in the shader** as
                                  RT_MASK/RT_SHIFT. Resizing it silently reads
                                  the wrong texels -- change both, keep it a
                                  power of two. Max Count is 256*256.
Count           float             MPC_MetaballConfig
TraceStart      float3            Camera Position
TraceDirection  float3            normalize(AbsoluteWorldPosition - CameraPosition), unit
WorldPosition   float3            Absolute World Position
RimThickness    float             contact-band width, in WORLD UNITS
```

Outputs: `return` → IsMetaSurface, `Normal` → MetaNormal, `Position` → MetaLocation.
There is **no rim output** — the contact rim is carried entirely by `Normal`, so the
node needs no Additional Output and the graph needs no extra pin.

`Spheres.a` carries the radius already squared and scaled, and `a > 0` doubles as
the visibility flag — so there is no separate radius texture, no per-ball squaring
in the shader, and no `.g` test. One fetch per ball.

`Locations`, `Radii`, `Scale`, `TraceStep`, `SceneDepth` and `CameraForward` are
all gone — if any are still wired, they are dead.

## How to run things

**Verify a shader change.** Mirror the edit into `harness.py` first, then:

```bash
py -3.10 harness.py --seed 4242
```

Vary the seed — one seed is not validation, and that is exactly how the
first-root bug survived. `--rays` and `--clip` are also available.

```bash
py -3.10 harness.py --seed 31337 --rays 6000
```

```bash
py -3.10 BezierMetaballsLab.py
```
```bash
py -3.10 BezierMetaballsLab.py --selftest
```

Compile-check the shader — it is a Custom node body, so wrap it in a function and
`#include` it:

```hlsl
Texture2D<float4> Spheres;

float MetaballTrace(float Count, float RimThickness,
                    float3 TraceStart, float3 TraceDirection, float3 WorldPosition,
                    inout float3 Normal, inout float3 Position)
{
#include "body.hlsl"
}
float4 MainPS(float4 svpos : SV_Position) : SV_Target
{
    float3 N = 0, P = 0;
    float3 dir = normalize(float3(svpos.xy * 0.001 - 0.5, 1.0));
    float hit = MetaballTrace(svpos.x, 8.0, float3(0,0,0), dir,
                              dir * (svpos.y + 100.0), N, P);
    return float4(N * hit, 1);
}
```

Keep `TraceDirection` varying per pixel and `WorldPosition` dependent on it. A
constant ray lets the compiler fold away large parts of what you are trying to
measure — it silently did exactly that once.

```bash
dxc -T ps_6_0 -E MainPS -O3 -Fc out.dxil wrapped.hlsl
```

`dxc`/`fxc` live in `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\`.

**Use `dxc`. The project targets SM6/D3D12, so DXIL is what ships.** `fxc` stays
useful for one thing — it prints `dcl_indexableTemp` sizes and per-loop instruction
counts, which DXIL does not — but treat those as a *relative* proxy for loop-body
size only. Its absolute numbers do not transfer, and on the question where the two
compilers disagree hardest, local arrays, `fxc` is simply wrong about the shipping
target. See "Things already ruled out".

It should compile with **no warnings**.

## The math

Wyvill's field, `u = (r/R)²`:

```
f(u) = -4/9 u³ + 17/9 u² - 22/9 u + 1
```

Along a ray, with `tmin`/`tmax` the chord ends:

```
w(t) = 1 - u = -(t - tmin)(t - tmax) / R²
f    = (5/9)w² + (4/9)w³            ← degree 6 in t
```

The root form is used rather than `-(t² + bt + c)/R²` because it avoids
large-number cancellation at world scale.

A lone ball's surface radius solves `(5/9)w² + (4/9)w³ = THRESHOLD`, then
`r = R·√(1-w)`. `bezier_metaballs_core.meta_radius(R, T)` does this.
**T = 0.148 → 0.746 R.**

## How the shader works

1. **Loader** — one fetch per ball from `Spheres`, ray/sphere test via the
   perpendicular offset, store `chords[k] = (tmin, tmax)` and
   `balls[k] = (centre.xyz, 1/R²)`. `chords` is a **float2** deliberately: the
   composite reads `1/R²` from `balls[j].w` instead, which is the same number of
   component loads and 128 bytes/thread cheaper. No sort. Runs `Count` times per pixel.
2. **Sweep** — cut the ray at every chord end. Between cuts the covering set is
   constant. The next cut is found inside the composite pass, so the array is
   walked once per segment, not twice.
3. **Composite, power basis** — each ball's density over a segment is a degree-6
   polynomial, so overlapping balls composite by **adding coefficients**. Exact,
   because segments are cut at chord ends and every active ball is fully "on".
4. **Bézier clipping** — convert to Bernstein, reject on the convex hull, bracket
   the first root between the steepest and shallowest control-point slopes out of
   `b0`, clip to that interval, repeat. Every bound is a hull bound, so the first
   root cannot be stepped over.

   The loop has **six exits, three accepting and three rejecting**, and `break`
   does not say which fired — that is what `alive` is for. The accepting exits are
   `F0 >= THRESHOLD`, `clip_len <= 1e-5`, and *running out of iterations*; the
   last one matters most, because `clip_p` is a monotone lower bound climbing onto
   the root, so an unconverged answer is early **within the correct segment**.
   Rejecting it would draw whatever lies behind that segment — a hole, not a small
   position error. Deleting `alive` and accepting unconditionally measured 19–27
   wrong hit/miss verdicts per 2330 rays and up to 11.9 world units of error,
   because a proven-root-free segment still exits with `clip_p = 0` and so reports
   a hit at `seg_p`.

   **`F0 >= THRESHOLD` must be tested before the `m_max <= 0` reject.** Merging
   them the other way round rejects a segment that starts inside the isosurface
   with the density falling outward, which must hit at `seg_p`: 124 such rays in
   2373, up to 9.7 world units of error — worse than having no `alive` at all.
   The upper bracket is the shallowest control-point slope, and when that lands
   outside the interval the **polygon peak is the only fallback probe**. A 4-probe
   uniform grid used to follow it; measured, the slope bound alone resolves 78% of
   iterations, the peak another 20%, and the grid ran on 2% and found anything on
   0.1%. Removing it was byte-identical on the harness across three seeds and took
   the clip loop from 163 instructions to 134. **Do not add a grid back** — a
   grazing hit can clear THRESHOLD by ~5e-4 across a fiftieth of a segment and an
   even grid straddles it; the peak is where the curve is highest. Failing to find
   an upper bound is not a failure: `t1` stays 1, the clip is left-ended only, and
   `clip_p` still rises onto the root.

   The Bernstein construction, the peak Horner and the Taylor shift are all
   `[unroll]` loops over `dens[7]` rather than hand-expanded arithmetic. The
   Bernstein coefficient needs no table because `C(j,k)/C(6,k)` telescopes to
   `a_k = (j-k+1)/(7-k)`, and at `k = j+1` that factor is exactly 0, so fxc folds
   each `b_j` back to its own `j+1` terms — **3 instructions cheaper** than
   writing the six control points out. The reassociation is not bit-identical to
   the explicit sum; it was checked in float32 across four seeds with no
   difference in any hit distance, and `harness._bernstein_explicit` is kept as a
   cross-check (max relative difference 3.8e-15 over 200k random coefficient
   sets).

   The two rejects may share one branch *below* the accept, since `hull_max >= F0`
   means the hull half cannot fire where the accept would have.
5. **Normal** — analytic field gradient, negated to point outward.

The two arrays are split **by consumer**, so every loop gets what it needs from a
single `float4` load: the sweep reads only `chords`, the normal loop and the
single-ball fast path read only `balls`. `1/R²` is duplicated between them
because both need it and the component exists either way.

**`entry = chords[k]` is load-bearing, not a readability temp** — it is the only
form that gets a single 128-bit indexable-temp read. Two separate things go wrong
without it: fxc does not CSE repeated reads at the same dynamic index, *and* a
component read costs an instruction each even when it does. Measured on the
composite loop, which runs ~22.8× per ray:

| form | scratch reads | body |
|---|---|---|
| `chords[k].x` etc. inline | 7 (`.x`×3, `.y`×3, `.z`) | 68 |
| three scalar locals | 3 | 57 |
| `float4 entry` | **1** (`.xxyz`) | **55** |

It has been deleted as dead weight once already.

## The contact rim

Where the isosurface meets scene geometry, a hard cut reads badly. Within
`RimThickness` **world units** of contact the normal is bent into the view plane,
so the band shades like a silhouette edge instead:

```
rim = saturate(1 - (scene_dist - surface) / RimThickness)
Nb  = N - rim · dot(N, D) · D          then renormalised
```

That scales out the component along the view ray: `rim = 1` leaves `N` exactly
perpendicular to `D`, `rim = 0` leaves it untouched, and everything between is
continuous. Rotating toward a chosen sideways direction would be undefined for a
camera-facing normal; this needs no such choice. `RimThickness = 0` disables it —
the divide goes to `+inf` and `1 - inf` saturates to 0. Square the rim term for a
softer falloff.

**There is no rim output.** The effect rides entirely on `Normal`, which is why
the node has no Additional Output. Both paths apply it, and they must stay
identical or the shading will pop as a second ball comes into range.

An earlier version emitted a separate `MeshRim` scalar measured in *pixels*, via
`ddx`/`ddy` of the view ray. That is gone — along with the requirement to take
derivatives in uniform control flow, which no longer applies anywhere in this
shader.

**`harness.py` does not cover this.** Every ray it traces uses
`scene_dist = 1e30`, so `rim` saturates to 0 and the bend is inert. There is also
no ground truth to check it against — the reference normal is the analytic field
gradient, which is precisely what the bend departs from. Only the render shows it.

One measured fact if you ever match a `dot(V, N)` silhouette rim to the same
width: that measure goes as √d, so screen-space distance-to-contour `|g|/|∇g|`
over-reports by exactly **2×** and needs a 0.5 factor.

## Occlusion — and the trap in it

```
float scene_dist = length(WorldPosition - TraceStart);
```

Three tests use it: the fast path (line ~172), a `break` inside the sweep
(line ~241), and a final check (line ~456). The contact rim reads it at two more.

This is correct **only because the material is post-process**. There,
`AbsoluteWorldPosition` is the world position of the *scene surface* at this pixel,
reconstructed from scene depth, so the length is the radial distance to the
occluder along this ray. In an ordinary mesh material the same node gives the
position of the surface being shaded — the hull the effect is drawn on — and the
shader would be comparing the isosurface against its own bounding geometry.

The sweep `break` is **purely an optimisation**. It is the only one that can abort
the walk before a surface is found, so a bad `scene_dist` shows up as holes rather
than as culling. That is what made the 2026-08-10 hunt so hard to read. Deleting it
changes no correct result; it only lets the sweep run further on occluded pixels.

**Anything that writes scene depth on top of the blobs will occlude them.** The
proxy sphere meshes on `BP_MetaballMaster` did exactly that: they write depth at
their own radius while the isosurface sits at `0.746·R` *inside* them, so each
metaball was occluded by its own proxy — marginally, which is why it appeared as
scattered dropped pixels rather than a clean disappearance. Hiding those components
fixed it. If the artifacts ever return, check what is in the depth buffer at the
blob positions first, before touching the algorithm.

## Conventions — get these right

- **Normals point OUTWARD**, on *both* paths. If the fast path and the sweep
  disagree, shading flips the moment a second ball comes into range. This has
  already happened once.
- **`TraceDirection` is unit** before it reaches the shader. Do not re-normalize.
- **Never drop a ball from the field to save work.** The density is the sum over
  `chords[]` and the normal is its gradient, so culling one bends the surface and
  the normal everywhere its influence reached. Occlusion applies to the *result*,
  never to the field.
- **Comments marked `DO NOT` are load-bearing.** Each marks a bug that shipped and
  was then found. They all look like tidy-ups.

## Performance

Headline figures: **563 DXIL instructions** (`dxc -T ps_6_0 -O3`) and, from RGA
on gfx1100, **176 VGPRs / 0 bytes scratch / 8 waves per SIMD**, since that is
what ships. The per-loop table is from `fxc`, because DXIL reports no such
breakdown — use it for *ratios between loops*, not absolute cost. Judge changes by
**loop body size × trip count**, never by any total — the total counts code, not trips, and it moved
the wrong way twice while the shader got faster.

| loop | body | trips |
|---|---|---|
| loader | 40 | **`Count` per pixel** |
| sweep | 268 | ~2.4 / ray |
| composite | 63 (~12 when skipping) | 22.8 / ray |
| clip | 167 | 3.03 / ray |
| normal | 18 | once |

**The array is REGISTERS, not scratch -- measure with RGA, not by counting bytes.**
On gfx1100 the compiler promotes the dynamically-indexed `chords[32]` into VGPRs
through indexed register addressing: RGA reports **0 bytes of scratch**. Every
array element therefore costs register file, and occupancy, directly. The DXBC
"indexable temp spills to memory" model is wrong for the shipping target.

```bash
rga -s dx12 --offline --ps wrapped.hlsl --ps-model ps_6_0 --ps-entry MainPS -c gfx1100 -a stats.csv
```

| build | VGPRs | waves/SIMD |
|---|---|---|
| `chords[32]` + `balls[32]` | 233 | 6 |
| **`chords[32]` only, centre re-fetched** | **176** | **8** |
| cap 16, both arrays | 144 | 10 |
| cap 8 | 89 | 17 |

Dropping `balls[]` was the free one: 128 VGPRs, read only twice per pixel and
never in the hot sweep, so re-fetching the centre from `Spheres` costs a cached,
wave-coherent load and buys two waves. The cap is the other lever and is NOT
free -- see Open items. The array is indexed by per-ray *intersections*, never by
scene `Count`.

The `next_q` reduction in both the loader and the composite exploits `tmin <= tmax`:
the earlier chord end still ahead is `tmin` if `tmin` clears the cutoff and `tmax`
otherwise, so two guarded `min`s collapse to one select and one `min`. In the loader
the second guard drops entirely, since `tmax > 0` is already proven by the `continue`
above it. Both were checked against the original over 400k random cases including
exact ties and tangent rays (`tmin == tmax`).

Storing centres rather than re-fetching them from `Spheres` is the trade behind that
second array: **470 → 457 slots and 3 → 1 texture fetches, for 512 → 1024 B/lane**.
It also deleted the 12:12 texel packing and its decode. Whether the extra 512 B costs
occupancy was never measured in-engine — if a future pass wants the bytes back, that
is the knob, and reverting it is mechanical.

**The biggest remaining lever is not in the shader.** The loader runs `Count` times
per pixel and dominates everything else at high ball counts — a pass that shaved 3
instructions off its body produced no measurable frame-time change. Merging the two
render targets into `Spheres` took it from two texture fetches per ball to one, and
the body from 61 instructions to 52; that was the last easy win. What remains is
algorithmic — see **culling** under Open items.

## Things already ruled out — do not re-litigate

- Position is **not** a world-space problem. `Position = TraceStart + surface*TraceDirection`
  is correct and `surface` was verified as a true world distance.
- `TraceDirection` normalization is **not** the issue; it is already unit.
- The power basis is **not** a precision problem at world scale, given the root
  form above.
- Narrowing array element types to save scratch does **not** work (vec4 granularity).
- **The compilers disagree about local arrays, and only `dxc` counts.** `dxc`
  scalarises a constant-indexed local array completely — `dens[7]` leaves no alloca in
  the DXIL and costs about nothing. `fxc` refuses even with entirely constant
  indices, allocating `x2[7]` at full float4 stride and calling it 37 slots dearer.
  Any DXBC measurement of a local array is misleading here. Dynamically indexed
  arrays (`chords[k]`, `balls[k]`) stay in memory under both — that is why
  `entry` still earns its place.
- Re-fetching centres from `Spheres` to save an array is **not** a win — it was
  measured both ways; see Performance.
- `[unroll]` on the clip loop is **not** a win: 2403 instruction slots against 484.
- The loader's depth cull (`tmin >= NEAREST_SURFACE`) **is** a win, but only
  because it sits BELOW the three earlier `continue`s. It then runs ~1.3 times
  per ray rather than `Count` times: 2.6 instructions against 139-255 saved,
  mostly by pushing 77-98% of rays to the `intersections == 0` early-out. An
  earlier measurement rejected it by costing the test at `2 x Count` — that was
  wrong, and the placement is the whole argument. Moving it above the `c <= 0`
  test would invert the economics.
- An optimisation pass in 2026-08 tried seven further candidates and **six lost**:
  deferring the loader's `sqrt` behind an algebraic behind-test (+3), the coverage
  test as one `min` (+1, and +5 on the clip loop), negating `det` once (+1),
  folding `seg_len` into the shifted roots (+1), carrying `g` with the opposite
  sign (−1 static but +5 on the clip loop, so worse per ray), and `rcp` in place of
  the bracket divides (+5). The compiler already performs all of these. Only moving
  the capacity test into the loader's loop condition won, at −2 on the hottest
  loop. Do not re-try the six without measuring.
- Writing `S` (and `wc`) as convolution loops, the way the `F` update is written,
  is **not** a win: +3 instructions for `S`. The explicit form exploits a symmetry
  the loop cannot — `2*wc0*wc1` is one product and a scale, where the convolution
  computes `wc0*wc1` and `wc1*wc0` separately and neither compiler folds them.
- Two later wins DID land, both in the loader: dropping the `center` temp for `fc`
  (register only, DXIL unchanged), and flipping `fc` to run **centre <- camera** so
  `dot(fc, D)` is already the negated projection and neither root needs a negate
  (585 -> 581). The fast path deliberately does NOT do this: it reuses `fc` as the
  perpendicular for the surface normal, so the sign would have to come back.
- A uniform probe grid beneath the polygon peak is **not** worth its 29 instructions;
  measured byte-identical without it. See the clip-loop notes above.
- `entry = chords[k]` is **not** a readability temp. See the array notes above.
- `SceneDepth / cos(theta)` is **not** needed. In a post-process material
  `AbsoluteWorldPosition` is already radial; that correction computed the same
  number the long way round.

## History worth knowing

Four bugs, none of which was found by reading the code.

1. **The root search could converge on a later root than the first.** The old
   probe-walk-plus-Newton search narrowed its bracket to the previous *probe*,
   which is not a proven bound, so a density bump between two probes put the first
   root outside the bracket — 2.0 world units of position error, 47.6° of normal
   error. Invisible because the harness only ever ran one RNG seed. Bézier clipping
   cannot do this by construction.

2. **The single-sphere path returned the view vector as its normal** whenever the
   camera was inside a ball, up to 176° wrong: `if (c <= 0) return 1;` fell through
   with `Normal` still at its initialised value. The harness skipped every
   single-ball ray, so the whole path was untested.

3. **Pinholes in distant isosurfaces**, from the ray/sphere discriminant.
   `det = b*b - 4*c` is catastrophic cancellation at range — both terms are ~4d²
   and their difference only ~4R². In float32 at d = 1e5 it corrupted 390 of 551
   chord ends *and invented 28 balls the ray never touched*; at d = 1e6 it returned
   exactly 0.0 and distant balls vanished. Replaced by
   `disc = R² - |f - (f·D)D|²`, which cancels nothing large — and which is also
   cheaper, since the roots come out `-h ∓ √disc` already ordered, so the divide,
   the `sign()` select and the `min`/`max` all disappear.

4. **Scattered dropped pixels on off-axis and distant metaballs** — the proxy
   sphere meshes writing scene depth over the blobs. See Occlusion above. This one
   cost the longest hunt by far, and the reason is worth recording: the shader was
   correct throughout, and every hypothesis that assumed otherwise was wrong. What
   finally located it was a diagnostic that *biased* `scene_dist` and showed the
   comparison was marginal rather than grossly wrong — not any amount of reasoning
   about the algorithm.

The lessons, in order of how much they cost:

- **Suspect the inputs before the algorithm.** The shader was innocent in (4), and
  a great deal of effort went into re-deriving parts that were already correct.
- **An untested path is where the bug is.** Two of the four lived in code the
  harness never executed.
- **A passing harness means the algorithm is right, not that the render is.**
  Nothing here can see the data that actually reaches the shader — not `Count`,
  not the contents of `Spheres`. When a clean harness and a broken render
  disagree, the harness is right and incomplete, and the inputs are worth
  dumping before the trace is touched.
- **The harness's float64 is not what ships.** (3) and the clip-loop convergence
  break were both float32-only.

## Open items

**Two ways to test, and they catch different things.** `harness.py` is a
*transliteration* — fast, easy to instrument, and able to drift from the shader
silently. The C++ path compiles `Metaballs.hlsl` itself behind a small shim
(`hlsl_compat.h`), so it cannot drift; the only preprocessing is stripping
`[unroll]`, which has no semantics. Cross-checked on 24000 identical rays they
agree exactly: 0 hit/miss disagreements, max `t` delta 4.2e-05, the `clip_len`
convergence bound. Iterate with the transliteration, confirm with the C++.

**`harness.py` is float64 and the shader is float32.** It says so in its own
header, and it is the one gap that has actually shipped a bug: (3) and the
clip-loop convergence break were both float32-only and the harness could not see
either. Anything touching the convergence break, the coefficient magnitudes, or a
subtraction of two large numbers needs a `np.float32` copy on top — NumPy ≥ 2
keeps float32 through arithmetic with Python literals, so a dtype-parameterised
copy of `trace` is enough. Verify any such copy reproduces the float64 version
exactly before trusting it: the first attempt at one used the wrong ball in the
fast path and invented 586 failures that did not exist.

**Culling is the only large win left.** The loader runs `Count` times per pixel and
dominates everything else at high ball counts — a pass that shaved 3 instructions
off its body produced no measurable frame-time change. Binning balls into screen
tiles so each pixel tests a handful instead of all of them changes the `Count` term
rather than the constant in front of it. Nothing in the shader does.

**The 32 cap is an occupancy/correctness trade, and both sides are now measured.**
Occupancy: 32 → 6 waves/SIMD, 16 → 10, 8 → 17 (RGA, gfx1100). Correctness:
intersections per ray, sampled across scene densities —

| scene | mean | p99 | max | over 16 |
|---|---|---|---|---|
| 300 balls, loose | 3.1 | 16 | 30 | 0.8% |
| 300 balls, tight | 23.1 | 90 | 126 | 49% |
| 800 balls, tight | 34.7 | 139 | 178 | 53% |
| dense chain | 93.2 | 249 | 269 | 65% |

In anything but a loose cluster the cap is ALREADY being hit. Overflow is silent
and keeps the first N in TEXTURE order, not the nearest N, so it pops with RT
write order. Shrinking the cap buys waves and costs dropped balls; tile culling
fixes both at once by making a generous cap affordable. Run `overflow.exe`
against the real ball distribution before touching it.

**These numbers are AMD.** The dev GPU is an RTX 3070, and NVIDIA has no indexed
register file, so there the arrays land in local memory instead — latency rather
than occupancy. UNVERIFIED: it needs the Vulkan SDK and a
`VK_KHR_pipeline_executable_properties` query.
