#include <math.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

typedef struct HitResult {
    int hit;
    int triangle_id;
    double distance;
} HitResult;

static int intersect_one(
    const double *triangle,
    double ox,
    double oy,
    double oz,
    double dx,
    double dy,
    double dz,
    double max_distance,
    double epsilon,
    double *out_distance
) {
    const double v0x = triangle[0];
    const double v0y = triangle[1];
    const double v0z = triangle[2];
    const double e1x = triangle[3];
    const double e1y = triangle[4];
    const double e1z = triangle[5];
    const double e2x = triangle[6];
    const double e2y = triangle[7];
    const double e2z = triangle[8];

    const double pvec_x = dy * e2z - dz * e2y;
    const double pvec_y = dz * e2x - dx * e2z;
    const double pvec_z = dx * e2y - dy * e2x;
    const double determinant = e1x * pvec_x + e1y * pvec_y + e1z * pvec_z;
    if (fabs(determinant) < epsilon) {
        return 0;
    }

    const double inv_det = 1.0 / determinant;
    const double tvec_x = ox - v0x;
    const double tvec_y = oy - v0y;
    const double tvec_z = oz - v0z;
    const double u = (tvec_x * pvec_x + tvec_y * pvec_y + tvec_z * pvec_z) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return 0;
    }

    const double qvec_x = tvec_y * e1z - tvec_z * e1y;
    const double qvec_y = tvec_z * e1x - tvec_x * e1z;
    const double qvec_z = tvec_x * e1y - tvec_y * e1x;
    const double v = (dx * qvec_x + dy * qvec_y + dz * qvec_z) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return 0;
    }

    const double distance = (e2x * qvec_x + e2y * qvec_y + e2z * qvec_z) * inv_det;
    if (distance <= epsilon || distance >= max_distance) {
        return 0;
    }

    *out_distance = distance;
    return 1;
}

EXPORT HitResult intersect_triangles(
    const double *triangles,
    int triangle_count,
    double ox,
    double oy,
    double oz,
    double dx,
    double dy,
    double dz,
    double max_distance,
    double epsilon
) {
    HitResult result;
    result.hit = 0;
    result.triangle_id = -1;
    result.distance = max_distance;

    for (int triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
        double distance = max_distance;
        const double *triangle = triangles + triangle_id * 9;
        if (intersect_one(triangle, ox, oy, oz, dx, dy, dz, result.distance, epsilon, &distance)) {
            result.hit = 1;
            result.triangle_id = triangle_id;
            result.distance = distance;
        }
    }

    return result;
}

EXPORT int is_occluded_triangles(
    const double *triangles,
    int triangle_count,
    double ox,
    double oy,
    double oz,
    double dx,
    double dy,
    double dz,
    double max_distance,
    double epsilon,
    int ignored_triangle_id
) {
    for (int triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
        if (triangle_id == ignored_triangle_id) {
            continue;
        }

        double distance = max_distance;
        const double *triangle = triangles + triangle_id * 9;
        if (intersect_one(triangle, ox, oy, oz, dx, dy, dz, max_distance, epsilon, &distance)) {
            return 1;
        }
    }

    return 0;
}
