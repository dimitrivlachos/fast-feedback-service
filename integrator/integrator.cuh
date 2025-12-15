/**
  * @file integrator.cuh
 */

#pragma once

#include <concepts>

/**
 * @brief Integer ceiling division
 * 
 * Computes ceil(n/d) using integer arithmetic only, avoiding
 * floating-point conversion. Equivalent to (n + d - 1) / d, which
 * rounds up to the nearest integer quotient.
 * 
 * @tparam T Integer type (constrained by std::integral concept)
 * @param n Numerator
 * @param d Denominator (must be > 0)
 * @return Ceiling of n/d
 * 
 * @example ceil_div(15, 4) returns 4, whereas 15/4 returns 3
 */
template <std::integral T>
constexpr inline T ceil_div(T n, T d) {
    return (n + d - 1) / d;
}

/**
 * @brief Process an image layer with GPU-accelerated Kabsch transformation
 * 
 * This function encapsulates all GPU kernel setup, memory management, and
 * kernel launch for processing reflections on a single image.
 * 
 * @param stream CUDA stream for async execution
 * @param device_image Pitched device memory containing the image
 * @param image_num Current image number
 * @param reflection_ids Reflection IDs that overlap this image
 * @param all_bboxes All bounding boxes (indexed by reflection ID)
 * @param s1_vectors All s1 vectors (indexed by reflection ID)
 * @param phi_positions All phi positions (indexed by reflection ID)
 * @param s0 Incident beam vector
 * @param rotation_axis Goniometer rotation axis
 * @param osc_start Oscillation start angle (radians)
 * @param osc_width Oscillation width per image (radians)
 * @param image_range_start First image number in the scan
 * @param wavelength Beam wavelength
 * @param pixel_size Detector pixel size [x, y] in mm
 * @param fast_axis Detector fast axis vector
 * @param slow_axis Detector slow axis vector
 * @param origin Detector origin position
 */
void call_do_integration(cudaStream_t stream,
                         PitchedMalloc<pixel_t> &device_image,
                         uint image_num,
                         const std::vector<size_t> &reflection_ids,
                         const std::vector<BoundingBoxExtents> &all_bboxes,
                         const std::vector<fastvec::Vector3D> &s1_vectors,
                         const std::vector<scalar_t> &phi_positions,
                         fastvec::Vector3D s0,
                         fastvec::Vector3D rotation_axis,
                         scalar_t osc_start,
                         scalar_t osc_width,
                         int image_range_start,
                         scalar_t wavelength,
                         const scalar_t pixel_size[2],
                         fastvec::Vector3D fast_axis,
                         fastvec::Vector3D slow_axis,
                         fastvec::Vector3D origin)