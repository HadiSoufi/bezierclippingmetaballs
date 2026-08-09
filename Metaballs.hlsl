if (Count == 0) return 0;

// Initialize inputs
Normal = normalize(TraceDirection);
Position = TraceStart;
Count = (int) Count;
float nearest_surface = dot(WorldPosition - TraceStart, WorldPosition - TraceStart);

// Load render target data
float w, h, x, y;
Locations.GetDimensions(w, h);

// Sphere data
float4 spheres[512];
float3 center;
float radius;

// Range data
int intersections = 0;
int j;
float3 fc;
float4 range;
float b, c, det, t0, t1;
bool not_in_sphere;

// x = range min, y = range max, z = index in spheres
float3 intersecting_spheres[512];

// x = range min, y = range max, z = left sphere index, w = right sphere index
float4 ranges[512];

// 1. Load sphere data from render target
// 2. Check to see if sphere is hit by this ray
// 3. Store sphere & intersection data
for (int i = 0; i < (int) Count; i++) {
    x = i % w;
    y = (int) i / h;

    // If sphere isn't visible (too distant or offscreen) skip
    // G is visibility flag, see BP_MetaballMaster
    if (!(Radii.Load(int3(x, y, 0)).g > 0)) continue;

    // Load from render targets
    center = Locations.Load(int3(x, y, 0)).rgb;
    radius = Radii.Load(int3(x, y, 0)).r * Scale;

    // Quadratic formula to get intersection data
    fc = TraceStart - center;
    b = 2 * dot(fc, TraceDirection);
    c = dot(fc, fc) - radius*radius;
    det = b*b - 4*c;
    not_in_sphere = c > 0;

    // b >= 0 is a backward or parallel ray- skip unless inside sphere
    // det < 0 means ray doesn't intersect this sphere
    if ((b >= 0 && not_in_sphere) || det < 0) continue;

    // Calculate intersections with ray as distances
    t0 = -(b + sign(b) * sqrt(det)) / 2;
    t1 = (t0 != 0) * c / t0;

    // Range from roots
    range = float4(not_in_sphere * min(t0, t1), max(t0, t1), i, 0);

    // Occlusion culling
    if(range.x * range.x > nearest_surface) continue;

    // Store sphere & range of overlap
    spheres[intersections] = float4(center.x, center.y, center.z, radius);
    intersecting_spheres[intersections] = range.xyz;

    // Store range, maintaining order via insertion sort
    range = float4(range.x, range.y, intersections, -1);
    for (j = intersections - 1; j >= 0 && ranges[j].x > range.x; j--) {
        ranges[j + 1] = ranges[j];
    }
    ranges[j + 1] = range;

    intersections++;
}

// Exit early if no intersections are found
if (intersections == 0) return 0;

// Special case- one intersected sphere
// So we can compute the surface analytically
if (intersections == 1) {
    // Computed by solving the metaball function for Threshold
    float meta_radius = 0.746;
    center = spheres[0].xyz;
    radius = spheres[0].w * meta_radius;

    // Quadratic again
    fc = TraceStart - center;
    b = 2 * dot(fc, TraceDirection);
    c = dot(fc, fc) - radius*radius;
    det = b*b - 4*c;

    // Succeed if camera inside sphere
    if (c <= 0) return 1;

    // Fail if trace not touching sphere
    if ((b >= 0 && c > 0) || det < 0) return 0;

    // Calculate roots
    t0 = -(b + sign(b) * sqrt(det)) / 2;
    t1 = (t0 != 0) * c / t0;

    // Find surface
    Position = min(t0, t1) * TraceDirection + TraceStart;

    // Fail if occlusion culling
    if (dot(TraceStart - Position, TraceStart - Position) >= nearest_surface) return 0;

    Normal = normalize(center - Position);
    return 1;
}

// Break overlapping ranges out into new arrays & update base ranges
float4 left_range, right_range;
float4 part_overlap_ranges[512], full_overlap_ranges[512];
int part_overlap_count = 0;
int full_overlap_count = 0;

for (int i = 0; i < intersections - 1; i++) {
    left_range = ranges[i];
    right_range = ranges[i + 1];

    // Full overlaps
    if (left_range.y > right_range.y) {
        ranges[i] = float4(left_range.x, right_range.x, left_range.z, -1);
        ranges[i + 1] = float4(right_range.y, left_range.y, left_range.z, -1);

        full_overlap_ranges[full_overlap_count] = float4(right_range.x, right_range.y, left_range.z, right_range.z);
        full_overlap_count++;

    // Partial overlaps
    } else if (left_range.y > right_range.x && right_range.y > left_range.y ) {
        ranges[i] = float4(left_range.x, right_range.x, left_range.z, -1);
        ranges[i + 1] = float4(left_range.y, right_range.y, right_range.z, -1);

        part_overlap_ranges[part_overlap_count] = float4(right_range.x, left_range.y, left_range.z, right_range.z);
        part_overlap_count++;
    }
}

bool found_surface = false;
float c2 = 0.0f, T = 0.0f, T2 = 0.0f, T3 = 0.0f, T4 = 0.0f, T5 = 0.0f, T6 = 0.0f;
float oneMinusT = 0.0f, oneMinusT2 = 0.0f, oneMinusT3 = 0.0f, oneMinusT4 = 0.0f, oneMinusT5 = 0.0f, oneMinusT6 = 0.0f;
float range_start = 0.0f, maxY = 0.0f, minY = 0.0f; // left_max, right_max;
float b_start = 0.0f, b_end = 0.0f, i_length = 0.0f;
int index = 0;

static float MAX_COEFF = 8.0 / 45.0;
static float MID_COEFF = 16.0 / 27.0;

float2 curve_template[7] = {float2(0.0, 0.0), float2(1.0 / 6.0, 0.0), float2(2.0 / 6.0, 0.0), float2(3.0 / 6.0, 0.0), float2(4.0 / 6.0, 0.0), float2(5.0 / 6.0, 0.0), float2(1.0, 0.0)};
float2 d[7], e[7];

// 0 = no overlaps, 1 = partial overlap, 2 = full overlap
float range_type = 0;

// Check un-overlapped ranges
for (int r = 0; r < intersections; r++) {
    index = ranges[r].z;
    radius = spheres[index].w;
    i_length = intersecting_spheres[index].y - intersecting_spheres[index].x;

    [unroll]
    for (int i = 0; i < 7; i++) {d[i] = curve_template[i];}

    c = i_length / (2 * radius);
    c *= c;
    c2 = c * c;

    maxY = MAX_COEFF * c2 * (8 * c + 5);
    minY = MID_COEFF * c2;

    d[2].y = minY;
    d[3].y = maxY;
    d[4].y = minY;

    if (maxY < Threshold) continue;

    found_surface = true;
    range_start = ranges[r].x;
    range_type = 0;
    break;
}

// Check partially overlapped ranges
for (int r = 0; r < part_overlap_count; r++) {
//     index = part_overlap_ranges[r].z;
//     radius = spheres[intersecting_spheres[index].z].w;
//     i_length = intersecting_spheres[index].y - intersecting_spheres[index].x;
//
//     c = i_length / (2 * radius);
//     c *= c;
//     c2 = c * c;
//
//     left_max = MAX_COEFF * c2 * (8 * c + 5);
//
//     index = part_overlap_ranges[r].w;
//     radius = spheres[intersecting_spheres[index].z].w;
//     i_length = intersecting_spheres[index].y - intersecting_spheres[index].x;
//
//     c = i_length / (2 * radius);
//     c *= c;
//     c2 = c * c;
//
//     right_max = MAX_COEFF * c2 * (8 * c + 5);
//
//     maxY = left_max + right_max;
//     found_surface = maxY >= Threshold || found_surface;
//
//     if (!found_surface) continue;

    [unroll]
    for (int i = 0; i < 7; i++) {
        d[i] = curve_template[i];
        e[i] = curve_template[i];
    }

    // Calculate density curve from left sphere
    index = part_overlap_ranges[r].z;
    radius = spheres[index].w;
    b_start = intersecting_spheres[index].x;
    b_end = intersecting_spheres[index].y;
    i_length = b_end - b_start;

    c = i_length / (radius * 2);
    c *= c;
    c2 = c * c;

    maxY = MAX_COEFF * c2 * (8 * c + 5);
    minY = MID_COEFF * c2;

    e[2].y = minY;
    e[3].y = maxY;
    e[4].y = minY;

    // Clip left sphere to the right half
    T  = (part_overlap_ranges[r].x - b_start) / i_length;
    T2 = T  * T;
    T3 = T2 * T;
    T4 = T3 * T;
    T5 = T4 * T;
    T6 = T5 * T;

    oneMinusT  = 1.0 - T;
    oneMinusT2 = oneMinusT  * oneMinusT;
    oneMinusT3 = oneMinusT2 * oneMinusT;
    oneMinusT4 = oneMinusT3 * oneMinusT;
    oneMinusT5 = oneMinusT4 * oneMinusT;

    d[0].y += 15 * T4 * oneMinusT2 * e[4].y + 20 * T3 * oneMinusT3 * e[3].y + 15 * T2 * oneMinusT4 * e[2].y;
    d[1].y += 10 * T3 * oneMinusT2 * e[4].y + 10 * T2 * oneMinusT3 * e[3].y + 5  * T  * oneMinusT4 * e[2].y;
    d[2].y += 6  * T2 * oneMinusT2 * e[4].y + 4  * T  * oneMinusT3 * e[3].y +           oneMinusT4 * e[2].y;
    d[3].y += 3  * T  * oneMinusT2 * e[4].y +           oneMinusT3 * e[3].y;
    d[4].y +=           oneMinusT2 * e[4].y;

    // ---- Early exit: curve starts above threshold at entry ----
    if (d[0].y >= Threshold) {
        Position = TraceStart + part_overlap_ranges[r].x * TraceDirection;
        Normal = normalize(spheres[index].xyz - Position);
        return 1;
    }

    // Calculate density curve for right sphere
    [unroll]
    for (int i = 0; i < 7; i++) {e[i] = curve_template[i];}

    index = part_overlap_ranges[r].w;
    radius = spheres[index].w;
    b_start = intersecting_spheres[index].x;
    b_end = intersecting_spheres[index].y;
    i_length = b_end - b_start;

    c = i_length / (radius * 2);
    c *= c;
    c2 = c * c;

    maxY = MAX_COEFF * c2 * (8 * c + 5);
    minY = MID_COEFF * c2;

    e[2].y = minY;
    e[3].y = maxY;
    e[4].y = minY;

    // Clip right sphere to left half
    T  = (part_overlap_ranges[r].y - b_start) / i_length;
    T2 = T  * T;
    T3 = T2 * T;
    T4 = T3 * T;
    T5 = T4 * T;
    T6 = T5 * T;

    oneMinusT  = 1.0 - T;
    oneMinusT2 = oneMinusT  * oneMinusT;
    oneMinusT3 = oneMinusT2 * oneMinusT;
    oneMinusT4 = oneMinusT3 * oneMinusT;
    oneMinusT5 = oneMinusT4 * oneMinusT;

    d[2].y +=                                                                               T2 *              e[2].y;
    d[3].y +=                                            T3 *              e[3].y + 3.0f  * T2 * oneMinusT  * e[2].y;
    d[4].y +=         T4 *              e[4].y + 4.0f  * T3 * oneMinusT  * e[3].y + 6.0f  * T2 * oneMinusT2 * e[2].y;
    d[5].y += 5.0f  * T4 * oneMinusT  * e[4].y + 10.0f * T3 * oneMinusT2 * e[3].y + 10.0f * T2 * oneMinusT3 * e[2].y;
    d[6].y += 15.0f * T4 * oneMinusT2 * e[4].y + 20.0f * T3 * oneMinusT3 * e[3].y + 15.0f * T2 * oneMinusT4 * e[2].y;

    maxY = d[0].y;
    minY = d[0].y;

    for(i = 0; i < 7; i++) {
        maxY = max(d[i].y, maxY);
        minY = min(d[i].y, minY);
    }

    if (maxY < Threshold) continue;

    found_surface = true;
    range_start = part_overlap_ranges[r].x;
    range_type = 1;
    break;
}

// Check fully overlapped ranges
for (int r = 0; r < full_overlap_count; r++) {

    index = full_overlap_ranges[r].z;
    radius = spheres[index].w;
    b_start = intersecting_spheres[index].x;
    b_end = intersecting_spheres[index].y;

    // ---- Build outer sphere's symmetric curve directly into d ----
    c = (b_end - b_start) / (2.0f * radius);
    c2 = c * c;
    d[0].y = 0.0f; d[1].y = 0.0f; d[5].y = 0.0f; d[6].y = 0.0f;
    d[2].y = MID_COEFF * c2;
    d[3].y = MAX_COEFF * c2 * (8.0f * c + 5.0f);
    d[4].y = d[2].y;
    
    [unroll]
    for (int i = 0; i < 7; i++) e[i] = d[i];

    // ---- 1. Clip RIGHT to match the end of the inner sphere ----
    float t_right = (full_overlap_ranges[r].y - b_start) / (b_end - b_start);
    T = clamp(t_right, 0.0f, 1.0f);
    T2 = T  * T; T3 = T2 * T; T4 = T3 * T; T5 = T4 * T; T6 = T5 * T;

    oneMinusT  = 1.0f - T;
    oneMinusT2 = oneMinusT  * oneMinusT;
    oneMinusT3 = oneMinusT2 * oneMinusT;
    oneMinusT4 = oneMinusT3 * oneMinusT;
    oneMinusT5 = oneMinusT4 * oneMinusT;
    oneMinusT6 = oneMinusT5 * oneMinusT;

    d[0].y = e[0].y;
    d[1].y = T * e[1].y + oneMinusT * e[0].y;
    d[2].y = T2 * e[2].y + 2.0f * T * oneMinusT * e[1].y + oneMinusT2 * e[0].y;
    d[3].y = T3 * e[3].y + 3.0f * T2 * oneMinusT * e[2].y + 3.0f * T * oneMinusT2 * e[1].y + oneMinusT3 * e[0].y;
    d[4].y = T4 * e[4].y + 4.0f * T3 * oneMinusT * e[3].y + 6.0f * T2 * oneMinusT2 * e[2].y + 4.0f * T * oneMinusT3 * e[1].y + oneMinusT4 * e[0].y;
    d[5].y = T5 * e[5].y + 5.0f * T4 * oneMinusT * e[4].y + 10.0f * T3 * oneMinusT2 * e[3].y + 10.0f * T2 * oneMinusT3 * e[2].y + 5.0f * T * oneMinusT4 * e[1].y + oneMinusT5 * e[0].y;
    d[6].y = T6 * e[6].y + 6.0f * T5 * oneMinusT * e[5].y + 15.0f * T4 * oneMinusT2 * e[4].y + 20.0f * T3 * oneMinusT3 * e[3].y + 15.0f * T2 * oneMinusT4 * e[2].y + 6.0f * T * oneMinusT5 * e[1].y + oneMinusT6 * e[0].y;

    [unroll]
    for (int i = 0; i < 7; i++) e[i] = d[i];

    // ---- 2. Clip LEFT to match the start of the inner sphere ----
    float t_left = (full_overlap_ranges[r].x - b_start) / (b_end - b_start);
    T = (t_right > 1e-5f) ? clamp(t_left / t_right, 0.0f, 1.0f) : 0.0f;
    T2 = T  * T; T3 = T2 * T; T4 = T3 * T; T5 = T4 * T; T6 = T5 * T;

    oneMinusT  = 1.0f - T;
    oneMinusT2 = oneMinusT  * oneMinusT;
    oneMinusT3 = oneMinusT2 * oneMinusT;
    oneMinusT4 = oneMinusT3 * oneMinusT;
    oneMinusT5 = oneMinusT4 * oneMinusT;
    oneMinusT6 = oneMinusT5 * oneMinusT;

    d[6].y = e[6].y;
    d[5].y = T * e[6].y + oneMinusT * e[5].y;
    d[4].y = T2 * e[6].y + 2.0f * T * oneMinusT * e[5].y + oneMinusT2 * e[4].y;
    d[3].y = T3 * e[6].y + 3.0f * T2 * oneMinusT * e[5].y + 3.0f * T * oneMinusT2 * e[4].y + oneMinusT3 * e[3].y;
    d[2].y = T4 * e[6].y + 4.0f * T3 * oneMinusT * e[5].y + 6.0f * T2 * oneMinusT2 * e[4].y + 4.0f * T * oneMinusT3 * e[3].y + oneMinusT4 * e[2].y;
    d[1].y = T5 * e[6].y + 5.0f * T4 * oneMinusT * e[5].y + 10.0f * T3 * oneMinusT2 * e[4].y + 10.0f * T2 * oneMinusT3 * e[3].y + 5.0f * T * oneMinusT4 * e[2].y + oneMinusT5 * e[1].y;
    d[0].y = T6 * e[6].y + 6.0f * T5 * oneMinusT * e[5].y + 15.0f * T4 * oneMinusT2 * e[4].y + 20.0f * T3 * oneMinusT3 * e[3].y + 15.0f * T2 * oneMinusT4 * e[2].y + 6.0f * T * oneMinusT5 * e[1].y + oneMinusT6 * e[0].y;
    
    // ---- 3. Add inner sphere's curve directly into d ----
    // Renamed local radius variable to prevent compiler shadowing bugs
    radius = spheres[full_overlap_ranges[r].w].w;
    c = (full_overlap_ranges[r].y - full_overlap_ranges[r].x) / (2.0f * radius);
    c2 = c * c;
    d[2].y += MID_COEFF * c2;
    d[3].y += MAX_COEFF * c2 * (8.0f * c + 5.0f);
    d[4].y += MID_COEFF * c2;

    maxY = d[0].y;
    minY = d[0].y;

    [unroll]
    for (int i = 1; i < 7; i++) {
        maxY = max(maxY, d[i].y);
        minY = min(minY, d[i].y);
    }

    if (maxY < Threshold) continue;

    found_surface = true;
    range_start = full_overlap_ranges[r].x;
    range_type = 2;
    break;
}

if (!found_surface) return 0;

// Bezier clipping to find surface
float t_max = 1.0f, t_min = 0.0f;
float g_offset = 0.0, g_ratio = 1.0;
float theta, start_min_theta, start_max_theta, end_min_theta, end_max_theta;
int clip_count = 0;

for (int i = 0; i < 20; i++) {
    clip_count++;
    // normalize x values & backup
    [unroll]
    for (int j = 0; j < 7; j++) {
        d[j].x = curve_template[j].x;
        e[j] = d[j];
    }

    // Clip to range. Left first
    T  = t_max;
    T2 = T  * T;
    T3 = T2 * T;
    T4 = T3 * T;
    T5 = T4 * T;
    T6 = T5 * T;

    oneMinusT  = 1.0 - T;
    oneMinusT2 = oneMinusT  * oneMinusT;
    oneMinusT3 = oneMinusT2 * oneMinusT;
    oneMinusT4 = oneMinusT3 * oneMinusT;
    oneMinusT5 = oneMinusT4 * oneMinusT;
    oneMinusT6 = oneMinusT5 * oneMinusT;

    d[0].y =      e[0].y;
    d[1].y = T  * e[1].y +             oneMinusT * e[0].y;
    d[2].y = T2 * e[2].y + 2.0f * T  * oneMinusT * e[1].y +              oneMinusT2 * e[0].y;
    d[3].y = T3 * e[3].y + 3.0f * T2 * oneMinusT * e[2].y + 3.0f  * T  * oneMinusT2 * e[1].y +              oneMinusT3 * e[0].y;
    d[4].y = T4 * e[4].y + 4.0f * T3 * oneMinusT * e[3].y + 6.0f  * T2 * oneMinusT2 * e[2].y + 4.0f  * T  * oneMinusT3 * e[1].y +              oneMinusT4 * e[0].y;
    d[5].y = T5 * e[5].y + 5.0f * T4 * oneMinusT * e[4].y + 10.0f * T3 * oneMinusT2 * e[3].y + 10.0f * T2 * oneMinusT3 * e[2].y + 5.0f  * T  * oneMinusT4 * e[1].y +            oneMinusT5 * e[0].y;
    d[6].y = T6 * e[6].y + 6.0f * T5 * oneMinusT * e[5].y + 15.0f * T4 * oneMinusT2 * e[4].y + 20.0f * T3 * oneMinusT3 * e[3].y + 15.0f * T2 * oneMinusT4 * e[2].y + 6.0f * T * oneMinusT5 * e[1].y + oneMinusT6 * e[0].y;

    // Find max & min, break if no intersection with Threshold
    maxY = d[0].y; minY = d[0].y;
    [unroll]
    for (int j = 0; j < 7; j++) {
        maxY = max(maxY, d[j].y);
        minY = min(minY, d[j].y);
    }

    // normalize x values & backup
    [unroll]
    for (int j = 0; j < 7; j++) {
        d[j].x = curve_template[j].x;
        e[j] = d[j];
    }

    // Now right of left
    T  = t_min / t_max;
    T2 = T  * T;
    T3 = T2 * T;
    T4 = T3 * T;
    T5 = T4 * T;
    T6 = T5 * T;

    oneMinusT  = 1.0 - T;
    oneMinusT2 = oneMinusT  * oneMinusT;
    oneMinusT3 = oneMinusT2 * oneMinusT;
    oneMinusT4 = oneMinusT3 * oneMinusT;
    oneMinusT5 = oneMinusT4 * oneMinusT;
    oneMinusT6 = oneMinusT5 * oneMinusT;

    d[6].y =      e[6].y;
    d[5].y = T  * e[6].y +            oneMinusT * e[5].y;
    d[4].y = T2 * e[6].y + 2.0 * T *  oneMinusT * e[5].y +             oneMinusT2 * e[4].y;
    d[3].y = T3 * e[6].y + 3.0 * T2 * oneMinusT * e[5].y + 3.0  * T  * oneMinusT2 * e[4].y +             oneMinusT3 * e[3].y;
    d[2].y = T4 * e[6].y + 4.0 * T3 * oneMinusT * e[5].y + 6.0  * T2 * oneMinusT2 * e[4].y + 4.0  * T  * oneMinusT3 * e[3].y +             oneMinusT4 * e[2].y;
    d[1].y = T5 * e[6].y + 5.0 * T4 * oneMinusT * e[5].y + 10.0 * T3 * oneMinusT2 * e[4].y + 10.0 * T2 * oneMinusT3 * e[3].y + 5.0  * T *  oneMinusT4 * e[2].y +           oneMinusT5 * e[1].y;
    d[0].y = T6 * e[6].y + 6.0 * T5 * oneMinusT * e[5].y + 15.0 * T4 * oneMinusT2 * e[4].y + 20.0 * T3 * oneMinusT3 * e[3].y + 15.0 * T2 * oneMinusT4 * e[2].y + 6.0 * T * oneMinusT5 * e[1].y + oneMinusT6 * e[0].y;

    // Find max & min, break if no intersection with Threshold
    maxY = d[0].y;
    minY = d[0].y;

    for (int j = 0; j < 7; j++) {
        maxY = max(maxY, d[j].y);
        minY = min(minY, d[j].y);
    }

    if(!(maxY > Threshold && minY < Threshold)) break;

    // Generate convex hull intersections

    // 1. Calculate the extreme slopes from the start and end points
    float start_min_theta = 1e10f;
    float start_max_theta = -1e10f;
    float end_min_theta = 1e10f;
    float end_max_theta = -1e10f;

    [unroll]
    for (int j = 1; j < 7; j++) {
        // ---- From the start point (d[0]) ----
        float dy_start = d[j].y - d[0].y;
        float dx_start = d[j].x - d[0].x; 
        float theta_start = dy_start / dx_start;
        
        start_min_theta = min(start_min_theta, theta_start);
        start_max_theta = max(start_max_theta, theta_start);

        // ---- From the end point (d[6]) ----
        float dy_end = d[6].y - d[j-1].y;
        float dx_end = d[6].x - d[j-1].x; 
        float theta_end = dy_end / dx_end;
        
        end_min_theta = min(end_min_theta, theta_end);
        end_max_theta = max(end_max_theta, theta_end);
    }

    // 2. Compute safe bounding intervals 
    float dy_start = Threshold - d[0].y;
    float t_min_new = 0.0f;

    if (abs(dy_start) > 1e-5f) {
        if (dy_start > 0.0f) {
            // Curve is below threshold, must go UP. Earliest hit is via the steepest positive slope.
            t_min_new = (start_max_theta > 1e-5f) ? (dy_start / start_max_theta) : 1.0f;
        } else {
            // Curve is above threshold, must go DOWN. Earliest hit is via steepest negative slope.
            t_min_new = (start_min_theta < -1e-5f) ? (dy_start / start_min_theta) : 1.0f;
        }
    }

    float dy_end = Threshold - d[6].y;
    float t_max_new = 1.0f;

    if (abs(dy_end) > 1e-5f) {
        if (dy_end > 0.0f) {
            // End point below threshold, looking backward curve must go DOWN. 
            // Latest hit is most negative backward slope.
            t_max_new = (end_min_theta < -1e-5f) ? (1.0f + dy_end / end_min_theta) : 0.0f;
        } else {
            // End point above threshold, looking backward curve must go UP. 
            // Latest hit is most positive backward slope.
            t_max_new = (end_max_theta > 1e-5f) ? (1.0f + dy_end / end_max_theta) : 0.0f;
        }
    }

    t_min = clamp(t_min_new, 0.0f, 1.0f);
    t_max = clamp(t_max_new, 0.0f, 1.0f);

    // 3. If interval is invalid, root is missed
    if (t_min >= t_max) break;

    // 4. Update the global offset and ratio for world-space position
    g_offset += t_min * g_ratio;
    g_ratio *= (t_max - t_min);

    // 5. Early exit if the root is found to prevent hovering when there are multiple roots
    if (abs(Threshold - d[0].y) < 0.001f) break; 

    // Break the loop if the interval is sufficiently small
    if (t_max - t_min < 0.02f) break;
}

if (t_min >= t_max) return 0;

float surface = g_offset + range_start;
Position = TraceStart + (surface * TraceDirection);

center = spheres[index].xyz;
Normal = normalize(center - Position);

//if(surface * surface >= nearest_surface) return 0;

center = spheres[index].xyz;
Normal = normalize(center - Position);
return 1;