/**
 * @file device_common.hu
 * @brief Common device functions
 */

#ifndef DEVICE_COMMON_H
#define DEVICE_COMMON_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

/**
 * @brief Struct to act as a global container for constant values
 * necessary for spotfinding
 */
struct ThresholdingConstants {
    size_t image_pitch;
    size_t mask_pitch;
    size_t result_pitch;
    int width;
    int height;
    float max_valid_pixel_value;
    uint8_t kernel_width;
    uint8_t kernel_height;
    uint8_t min_count;
    float n_sig_b;
    float n_sig_s;
};

#endif  // DEVICE_COMMON_H