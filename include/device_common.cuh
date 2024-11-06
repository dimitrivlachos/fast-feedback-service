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
 * @brief Load the halo pixels into shared memory.
 * 
 * This function loads a region of pixels into shared memory, including
 * the kernel halo region surrounding it.
 * 
 * @warning This function does not synchronize threads.
 *
 * @param image Pointer to the main image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param shared_image Pointer to the shared memory for the image.
 * @param shared_mask Pointer to the shared memory for the mask.
 * @param block The cooperative group representing the current block.
 * @param x The x-coordinate of the pixel in the global memory.
 * @param y The y-coordinate of the pixel in the global memory.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param image_pitch The pitch (width in bytes) of the image data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param kernel_width The halo radius in the x-direction.
 * @param kernel_height The halo radius in the y-direction.
 */
template <typename T>
__device__ void load_shared_memory_halo(T *image,
                                        uint8_t *mask,
                                        T *shared_image,
                                        uint8_t *shared_mask,
                                        cooperative_groups::thread_block block,
                                        int x,
                                        int y,
                                        size_t image_pitch,
                                        size_t mask_pitch,
                                        int width,
                                        int height,
                                        uint8_t kernel_width,
                                        uint8_t kernel_height) {
    // Compute local shared memory coordinates
    int local_x = threadIdx.x + kernel_width;
    int local_y = threadIdx.y + kernel_height;

    int shared_width = blockDim.x + 2 * kernel_width;

    // Load central pixel into shared memory
    shared_image[local_y * shared_width + local_x] = image[y * image_pitch + x];
    shared_mask[local_y * shared_width + local_x] = mask[y * mask_pitch + x];

    // Boundary checks for halo loading - only load surrounding pixels
    if (threadIdx.x < kernel_width && x >= kernel_width) {
        shared_image[local_y * shared_width + (local_x - kernel_width)] =
          image[y * image_pitch + (x - kernel_width)];
        shared_mask[local_y * shared_width + (local_x - kernel_width)] =
          mask[y * mask_pitch + (x - kernel_width)];
    }
    if (threadIdx.x >= blockDim.x - kernel_width && x + kernel_width < width) {
        shared_image[local_y * shared_width + (local_x + kernel_width)] =
          image[y * image_pitch + (x + kernel_width)];
        shared_mask[local_y * shared_width + (local_x + kernel_width)] =
          mask[y * mask_pitch + (x + kernel_width)];
    }
    if (threadIdx.y < kernel_height && y >= kernel_height) {
        shared_image[(local_y - kernel_height) * shared_width + local_x] =
          image[(y - kernel_height) * image_pitch + x];
        shared_mask[(local_y - kernel_height) * shared_width + local_x] =
          mask[(y - kernel_height) * mask_pitch + x];
    }
    if (threadIdx.y >= blockDim.y - kernel_height && y + kernel_height < height) {
        shared_image[(local_y + kernel_height) * shared_width + local_x] =
          image[(y + kernel_height) * image_pitch + x];
        shared_mask[(local_y + kernel_height) * shared_width + local_x] =
          mask[(y + kernel_height) * mask_pitch + x];
    }

    // Load corner pixels if within bounds
    if (threadIdx.x < kernel_width && threadIdx.y < kernel_height && x >= kernel_width
        && y >= kernel_height) {
        shared_image[(local_y - kernel_height) * shared_width
                     + (local_x - kernel_width)] =
          image[(y - kernel_height) * image_pitch + (x - kernel_width)];
        shared_mask[(local_y - kernel_height) * shared_width
                    + (local_x - kernel_width)] =
          mask[(y - kernel_height) * mask_pitch + (x - kernel_width)];
    }
    if (threadIdx.x >= blockDim.x - kernel_width && threadIdx.y < kernel_height
        && x + kernel_width < width && y >= kernel_height) {
        shared_image[(local_y - kernel_height) * shared_width
                     + (local_x + kernel_width)] =
          image[(y - kernel_height) * image_pitch + (x + kernel_width)];
        shared_mask[(local_y - kernel_height) * shared_width
                    + (local_x + kernel_width)] =
          mask[(y - kernel_height) * mask_pitch + (x + kernel_width)];
    }
    if (threadIdx.x < kernel_width && threadIdx.y >= blockDim.y - kernel_height
        && x >= kernel_width && y + kernel_height < height) {
        shared_image[(local_y + kernel_height) * shared_width
                     + (local_x - kernel_width)] =
          image[(y + kernel_height) * image_pitch + (x - kernel_width)];
        shared_mask[(local_y + kernel_height) * shared_width
                    + (local_x - kernel_width)] =
          mask[(y + kernel_height) * mask_pitch + (x - kernel_width)];
    }
    if (threadIdx.x >= blockDim.x - kernel_width
        && threadIdx.y >= blockDim.y - kernel_height && x + kernel_width < width
        && y + kernel_height < height) {
        shared_image[(local_y + kernel_height) * shared_width
                     + (local_x + kernel_width)] =
          image[(y + kernel_height) * image_pitch + (x + kernel_width)];
        shared_mask[(local_y + kernel_height) * shared_width
                    + (local_x + kernel_width)] =
          mask[(y + kernel_height) * mask_pitch + (x + kernel_width)];
    }
}

#endif  // DEVICE_COMMON_H