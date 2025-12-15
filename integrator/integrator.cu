/**
  * @file integrator.cu
 */

#include <memory>

#include "cuda_common.hpp"
#include "extent.hpp"
#include "integrator.cuh"
#include "kabsch.cuh"
#include "math/vector3d.cuh"

using CoordKey = std::pair<int, int>;
using KernelConfig = std::pair<dim3, dim3>;

/**
 * @brief GPU-specific metadata for mapping blocks to bounding boxes
 * 
 * Extended to include reflection parameters to avoid repeated lookups.
 */
struct BBoxGPUMetadata {
    int x_min;               ///< Minimum x pixel coordinate
    int y_min;               ///< Minimum y pixel coordinate
    int width;               ///< Width in pixels
    int height;              ///< Height in pixels
    int block_offset;        ///< First block index in 1D grid
    int blocks_x;            ///< Number of blocks in x dimension
    size_t refl_id;          ///< Reflection ID this bbox belongs to
    fastvec::Vector3D s1_c;  ///< Reflection center s1 vector (constant per bbox)
    scalar_t phi_c;          ///< Reflection center phi angle (constant per bbox)
};

/**
 * @brief Calculate grid configuration and create GPU metadata for an image layer
 * 
 * @param reflection_ids Reflection IDs that overlap this image
 * @param all_bboxes All bounding boxes (indexed by reflection ID)
 * @param s1_vectors All s1 vectors (indexed by reflection ID)
 * @param phi_positions All phi positions (indexed by reflection ID)
 * @param image_num Current image number
 * @param block_size Thread block size (default: 16 for 16x16 blocks)
 * @return Tuple of (grid_dim, block_dim, vector of GPU metadata)
 */
std::tuple<dim3, dim3, std::vector<BBoxGPUMetadata>> calculate_layer_config(
  const std::vector<size_t> &reflection_ids,
  const std::vector<BoundingBoxExtents> &all_bboxes,
  const std::vector<fastvec::Vector3D> &s1_vectors,
  const std::vector<scalar_t> &phi_positions,
  int image_num,
  int block_size = 16) {
    dim3 block_dim(block_size, block_size, 1);

    std::vector<BBoxGPUMetadata> metadata;
    metadata.reserve(reflection_ids.size());

    int total_blocks = 0;

    for (size_t refl_id : reflection_ids) {
        const auto &bbox = all_bboxes[refl_id];

        // Skip if this reflection doesn't overlap this image
        if (image_num < bbox.z_min || image_num > bbox.z_max) {
            continue;
        }

        BBoxGPUMetadata meta;
        meta.x_min = static_cast<int>(bbox.x_min);
        meta.y_min = static_cast<int>(bbox.y_min);
        meta.width = static_cast<int>(bbox.x_max - bbox.x_min);
        meta.height = static_cast<int>(bbox.y_max - bbox.y_min);
        meta.refl_id = refl_id;

        // Store reflection parameters directly in metadata
        meta.s1_c = s1_vectors[refl_id];
        meta.phi_c = phi_positions[refl_id];

        // Calculate blocks needed for this bbox (processing corners)
        // Pixels share corners with neighbors, so width x height pixels have (width+1) x (height+1) unique corners
        // Shift back by 1 to start at the first corner: x_min-1, y_min-1
        int corner_width = meta.width + 1;
        int corner_height = meta.height + 1;
        meta.blocks_x = ceil_div(corner_width, block_size);
        int blocks_y = ceil_div(corner_height, block_size);

        // Assign starting block index
        meta.block_offset = total_blocks;
        total_blocks += meta.blocks_x * blocks_y;

        metadata.push_back(meta);
    }

    dim3 grid_dim(total_blocks, 1, 1);
    return {grid_dim, block_dim, metadata};
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
                         fastvec::Vector3D origin) {
    // Calculate grid configuration and create GPU metadata
    constexpr int BLOCK_SIZE = 16;
    auto [grid_dim, block_dim, bbox_metadata] = calculate_layer_config(
      reflection_ids, all_bboxes, s1_vectors, phi_positions, image_num, BLOCK_SIZE);

    if (bbox_metadata.empty()) {
        return;  // No reflections on this image
    }

    // Copy metadata to device
    DeviceBuffer<BBoxGPUMetadata> d_metadata(bbox_metadata.size());
    d_metadata.assign(bbox_metadata.data());

    // Calculate phi angle for this image
    scalar_t phi_image = osc_start + (image_num - image_range_start + 0.5f) * osc_width;

    // Process each bbox separately
    std::vector<std::unique_ptr<DeviceBuffer<fastvec::Vector3D>>> eps_buffers;
    std::vector<std::unique_ptr<DeviceBuffer<scalar_t>>> s1_len_buffers;

    for (const auto &meta : bbox_metadata) {
        // Pixels share corners with neighbors: width x height pixels have (width+1) x (height+1) unique corners
        size_t num_corners = (meta.width + 1) * (meta.height + 1);

        // Compute s_pixels on CPU for this bbox
        std::vector<fastvec::Vector3D> h_s_pixels(num_corners);
        std::vector<scalar_t> h_phi_pixels(num_corners, phi_image);

        // Mayhaps this could be a kernel?

        for (int local_y = 0; local_y < meta.height + 1; ++local_y) {
            for (int local_x = 0; local_x < meta.width + 1; ++local_x) {
                // Calculate corner position offset back by 1
                scalar_t corner_x = meta.x_min - 1 + local_x;
                scalar_t corner_y = meta.y_min - 1 + local_y;

                // Convert to lab frame
                fastvec::Vector3D lab_pos = origin
                                            + fast_axis * (corner_x * pixel_size[0])
                                            + slow_axis * (corner_y * pixel_size[1]);

                // Convert to reciprocal space
                h_s_pixels[local_y * (meta.width + 1) + local_x] =
                  fastvec::normalized(lab_pos) / wavelength;
            }
        }

        // Allocate device buffers
        DeviceBuffer<fastvec::Vector3D> d_s_pixels(num_corners);
        DeviceBuffer<scalar_t> d_phi_pixels(num_corners);
        auto eps_buf = std::make_unique<DeviceBuffer<fastvec::Vector3D>>(num_corners);
        auto s1_len_buf = std::make_unique<DeviceBuffer<scalar_t>>(num_corners);

        // Copy to device
        d_s_pixels.assign(h_s_pixels.data());
        d_phi_pixels.assign(h_phi_pixels.data());

        // Configure and launch kabsch_transform kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_corners + threadsPerBlock - 1) / threadsPerBlock;

        // Launch existing kabsch_transform kernel
        kabsch_transform<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
          d_s_pixels.data(),
          d_phi_pixels.data(),
          meta.s1_c,
          meta.phi_c,
          s0,
          rotation_axis,
          eps_buf->data(),
          s1_len_buf->data(),
          num_corners);

        eps_buffers.push_back(std::move(eps_buf));
        s1_len_buffers.push_back(std::move(s1_len_buf));
    }

    cuda_throw_error();
}