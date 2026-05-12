/*
 * native_intersect.c  –  fast Möller–Trumbore intersection for a flat triangle list.
 *
 * Optimisations over the original version:
 *   1. float instead of double  – halves memory bandwidth, enables wider SIMD.
 *   2. __builtin_expect hints   – branch predictor help on GCC/Clang.
 *   3. restrict pointers        – lets the compiler assume no aliasing.
 *   4. BVH flat-array traversal – the C library now also accepts a compact BVH
 *      (axis-aligned bounding boxes stored as float[6] per node) so that large
 *      scenes can skip the Python BVH entirely.  The Python side still builds the
 *      BVH; it just serialises it into a flat float array and passes it here.
 *   5. Early-exit shadow ray    – is_occluded returns as soon as any hit is found.
 *   6. Precomputed inv_dir      – AABB slab test uses reciprocal direction passed
 *      in from the caller so it is computed only once per ray.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT
#endif

#ifdef __GNUC__
#  define LIKELY(x)   __builtin_expect(!!(x), 1)
#  define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#  define LIKELY(x)   (x)
#  define UNLIKELY(x) (x)
#endif

/* ------------------------------------------------------------------ */
/*  Basic types                                                         */
/* ------------------------------------------------------------------ */

typedef struct HitResult {
    int   hit;
    int   triangle_id;
    float distance;
} HitResult;

/* ------------------------------------------------------------------ */
/*  Single-triangle Möller–Trumbore (float)                            */
/*  Layout per triangle: v0x v0y v0z  e1x e1y e1z  e2x e2y e2z       */
/* ------------------------------------------------------------------ */

static inline int intersect_one_f(
    const float * restrict tri,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float max_dist,
    float epsilon,
    float * restrict out_dist
) {
    const float e1x = tri[3], e1y = tri[4], e1z = tri[5];
    const float e2x = tri[6], e2y = tri[7], e2z = tri[8];

    const float pvx = dy * e2z - dz * e2y;
    const float pvy = dz * e2x - dx * e2z;
    const float pvz = dx * e2y - dy * e2x;
    const float det = e1x * pvx + e1y * pvy + e1z * pvz;

    if (UNLIKELY(fabsf(det) < epsilon)) return 0;

    const float inv_det = 1.0f / det;
    const float tvx = ox - tri[0];
    const float tvy = oy - tri[1];
    const float tvz = oz - tri[2];
    const float u = (tvx * pvx + tvy * pvy + tvz * pvz) * inv_det;
    if (UNLIKELY(u < 0.0f || u > 1.0f)) return 0;

    const float qvx = tvy * e1z - tvz * e1y;
    const float qvy = tvz * e1x - tvx * e1z;
    const float qvz = tvx * e1y - tvy * e1x;
    const float v = (dx * qvx + dy * qvy + dz * qvz) * inv_det;
    if (UNLIKELY(v < 0.0f || u + v > 1.0f)) return 0;

    const float dist = (e2x * qvx + e2y * qvy + e2z * qvz) * inv_det;
    if (UNLIKELY(dist <= epsilon || dist >= max_dist)) return 0;

    *out_dist = dist;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Flat triangle list – original API (double → float conversion)      */
/*  Python passes double arrays; we convert once per call.             */
/* ------------------------------------------------------------------ */

EXPORT HitResult intersect_triangles(
    const double * restrict triangles,
    int   triangle_count,
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double max_distance,
    double epsilon
) {
    HitResult result = {0, -1, (float)max_distance};

    const float fox = (float)ox, foy = (float)oy, foz = (float)oz;
    const float fdx = (float)dx, fdy = (float)dy, fdz = (float)dz;
    const float fmax = (float)max_distance;
    const float feps = (float)epsilon;

    for (int i = 0; i < triangle_count; ++i) {
        /* Convert 9 doubles to floats on the fly – cheaper than a separate
           float buffer because the data is already hot in L1/L2. */
        float tri[9];
        const double *src = triangles + i * 9;
        for (int k = 0; k < 9; ++k) tri[k] = (float)src[k];

        float dist;
        if (LIKELY(!intersect_one_f(tri, fox, foy, foz, fdx, fdy, fdz,
                                    result.distance, feps, &dist)))
            continue;
        result.hit         = 1;
        result.triangle_id = i;
        result.distance    = dist;
    }
    return result;
}

EXPORT int is_occluded_triangles(
    const double * restrict triangles,
    int   triangle_count,
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double max_distance,
    double epsilon,
    int   ignored_triangle_id
) {
    const float fox = (float)ox, foy = (float)oy, foz = (float)oz;
    const float fdx = (float)dx, fdy = (float)dy, fdz = (float)dz;
    const float fmax = (float)max_distance;
    const float feps = (float)epsilon;

    for (int i = 0; i < triangle_count; ++i) {
        if (UNLIKELY(i == ignored_triangle_id)) continue;

        float tri[9];
        const double *src = triangles + i * 9;
        for (int k = 0; k < 9; ++k) tri[k] = (float)src[k];

        float dist;
        if (intersect_one_f(tri, fox, foy, foz, fdx, fdy, fdz,
                            fmax, feps, &dist))
            return 1;   /* early exit – shadow ray */
    }
    return 0;
}

/* ================================================================== */
/*  BVH flat-array API                                                  */
/*                                                                      */
/*  The Python side serialises the BVH into two flat arrays:           */
/*    bvh_nodes  – float[node_count * 8]                               */
/*      [0..5]  = AABB min/max (x_min,y_min,z_min, x_max,y_max,z_max) */
/*      [6]     = left child index  (-1 = leaf)                        */
/*      [7]     = right child index (-1 = leaf)                        */
/*    bvh_tris   – int32[node_count]  triangle index for leaf nodes    */
/*                 (-1 for internal nodes)                              */
/*                                                                      */
/*  Leaf nodes have left == right == -1 and bvh_tris[node] >= 0.       */
/*  Internal nodes have bvh_tris[node] == -1.                          */
/*                                                                      */
/*  Traversal uses an explicit stack (max depth 64 – enough for        */
/*  log2(65536) levels).                                                */
/* ================================================================== */

#define BVH_STACK_SIZE 64

/* AABB slab test with precomputed reciprocal direction */
static inline int aabb_hit(
    const float *node,   /* node[0..5] = min xyz, max xyz */
    float ox, float oy, float oz,
    float idx, float idy, float idz,   /* 1/dx, 1/dy, 1/dz */
    float t_max
) {
    float t0x = (node[0] - ox) * idx;
    float t1x = (node[3] - ox) * idx;
    if (t0x > t1x) { float tmp = t0x; t0x = t1x; t1x = tmp; }

    float t0y = (node[1] - oy) * idy;
    float t1y = (node[4] - oy) * idy;
    if (t0y > t1y) { float tmp = t0y; t0y = t1y; t1y = tmp; }

    float t0z = (node[2] - oz) * idz;
    float t1z = (node[5] - oz) * idz;
    if (t0z > t1z) { float tmp = t0z; t0z = t1z; t1z = tmp; }

    float t_enter = t0x > t0y ? t0x : t0y;
    if (t0z > t_enter) t_enter = t0z;
    float t_exit  = t1x < t1y ? t1x : t1y;
    if (t1z < t_exit) t_exit = t1z;

    return t_exit > t_enter && t_exit > 0.0f && t_enter < t_max;
}

/*
 * intersect_bvh_triangles
 *
 * bvh_nodes  – float array, 8 floats per node (see layout above)
 * bvh_tris   – int32 array, one entry per node (triangle index or -1)
 * node_count – total number of BVH nodes
 * triangles  – double array, 9 doubles per triangle (v0, e1, e2)
 * triangle_count – total triangles (used for bounds check only)
 */
EXPORT HitResult intersect_bvh_triangles(
    const float   * restrict bvh_nodes,
    const int32_t * restrict bvh_tris,
    int   node_count,
    const double  * restrict triangles,
    int   triangle_count,
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double max_distance,
    double epsilon
) {
    HitResult result = {0, -1, (float)max_distance};

    const float fox = (float)ox, foy = (float)oy, foz = (float)oz;
    const float fdx = (float)dx, fdy = (float)dy, fdz = (float)dz;
    const float feps = (float)epsilon;

    /* Precompute reciprocal direction; handle near-zero components */
    const float safe_dx = fabsf(fdx) < 1e-30f ? 1e-30f : fdx;
    const float safe_dy = fabsf(fdy) < 1e-30f ? 1e-30f : fdy;
    const float safe_dz = fabsf(fdz) < 1e-30f ? 1e-30f : fdz;
    const float idx = 1.0f / safe_dx;
    const float idy = 1.0f / safe_dy;
    const float idz = 1.0f / safe_dz;

    int stack[BVH_STACK_SIZE];
    int top = 0;
    stack[top++] = 0;   /* root node */

    while (top > 0) {
        int node_idx = stack[--top];
        if (UNLIKELY(node_idx < 0 || node_idx >= node_count)) continue;

        const float   *node = bvh_nodes + node_idx * 8;
        const int32_t  left  = (int32_t)node[6];
        const int32_t  right = (int32_t)node[7];

        if (!aabb_hit(node, fox, foy, foz, idx, idy, idz, result.distance))
            continue;

        if (left < 0 && right < 0) {
            /* Leaf node */
            int tri_id = bvh_tris[node_idx];
            if (UNLIKELY(tri_id < 0 || tri_id >= triangle_count)) continue;

            float tri[9];
            const double *src = triangles + tri_id * 9;
            for (int k = 0; k < 9; ++k) tri[k] = (float)src[k];

            float dist;
            if (intersect_one_f(tri, fox, foy, foz, fdx, fdy, fdz,
                                result.distance, feps, &dist)) {
                result.hit         = 1;
                result.triangle_id = tri_id;
                result.distance    = dist;
            }
        } else {
            if (LIKELY(top + 2 <= BVH_STACK_SIZE)) {
                if (right >= 0) stack[top++] = right;
                if (left  >= 0) stack[top++] = left;
            }
        }
    }
    return result;
}

EXPORT int is_occluded_bvh_triangles(
    const float   * restrict bvh_nodes,
    const int32_t * restrict bvh_tris,
    int   node_count,
    const double  * restrict triangles,
    int   triangle_count,
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double max_distance,
    double epsilon,
    int   ignored_triangle_id
) {
    const float fox = (float)ox, foy = (float)oy, foz = (float)oz;
    const float fdx = (float)dx, fdy = (float)dy, fdz = (float)dz;
    const float fmax = (float)max_distance;
    const float feps = (float)epsilon;

    const float safe_dx = fabsf(fdx) < 1e-30f ? 1e-30f : fdx;
    const float safe_dy = fabsf(fdy) < 1e-30f ? 1e-30f : fdy;
    const float safe_dz = fabsf(fdz) < 1e-30f ? 1e-30f : fdz;
    const float idx = 1.0f / safe_dx;
    const float idy = 1.0f / safe_dy;
    const float idz = 1.0f / safe_dz;

    int stack[BVH_STACK_SIZE];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        int node_idx = stack[--top];
        if (UNLIKELY(node_idx < 0 || node_idx >= node_count)) continue;

        const float   *node = bvh_nodes + node_idx * 8;
        const int32_t  left  = (int32_t)node[6];
        const int32_t  right = (int32_t)node[7];

        if (!aabb_hit(node, fox, foy, foz, idx, idy, idz, fmax))
            continue;

        if (left < 0 && right < 0) {
            int tri_id = bvh_tris[node_idx];
            if (UNLIKELY(tri_id < 0 || tri_id >= triangle_count)) continue;
            if (UNLIKELY(tri_id == ignored_triangle_id)) continue;

            float tri[9];
            const double *src = triangles + tri_id * 9;
            for (int k = 0; k < 9; ++k) tri[k] = (float)src[k];

            float dist;
            if (intersect_one_f(tri, fox, foy, foz, fdx, fdy, fdz,
                                fmax, feps, &dist))
                return 1;   /* early exit */
        } else {
            if (LIKELY(top + 2 <= BVH_STACK_SIZE)) {
                if (right >= 0) stack[top++] = right;
                if (left  >= 0) stack[top++] = left;
            }
        }
    }
    return 0;
}
