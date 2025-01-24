#include "connected_components.hpp"

#include <builtin_types.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <cstdint>
#include <vector>

#include "common.hpp"
#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"

#pragma region Connected Components
ConnectedComponents::ConnectedComponents(const uint8_t *result_image,
                                         const pixel_t *original_image,
                                         const ushort width,
                                         const ushort height,
                                         const uint min_spot_size)
    : num_strong_pixels(0), num_strong_pixels_filtered(0) {
    // Construct signals
    size_t k = 0;  // Linear index for the image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++k) {
            if (result_image[k]) {  // Store only non-zero (signal) pixels
                signals[k] = {x, y, std::nullopt, original_image[k], k};
                ++num_strong_pixels;
            }
        }
    }

    // Build graph
    build_graph(width, height);

    // Generate bounding boxes
    generate_boxes(width, height, min_spot_size);
}

/**
 * Build a graph from the pixel coordinates
 * 
 * The graph is built by iterating over the pixel coordinates and connecting
 * pixels that are adjacent to each other.
 */
void ConnectedComponents::build_graph(const ushort width, const ushort height) {
    size_t vertex_id = 0;

    // First, add vertices to the graph
    for (const auto &[linear_index, signal] : signals) {
        // Store the mappings for the linear index to vertex ID and vice versa
        index_to_vertex[linear_index] = vertex_id;
        vertex_to_index[vertex_id] = linear_index;
        ++vertex_id;

        boost::add_vertex(graph);
    }

    // Add edges by checking neighbors
    for (const auto &[linear_index, signal] : signals) {
        size_t right_linear_index = linear_index + 1;      // Pixel to the right
        size_t below_linear_index = linear_index + width;  // Pixel below

        // Check and connect to the pixel on the right
        if (signals.find(right_linear_index) != signals.end()) {
            boost::add_edge(index_to_vertex[linear_index],
                            index_to_vertex[right_linear_index],
                            graph);
        }

        // Check and connect to the pixel below
        if (signals.find(below_linear_index) != signals.end()) {
            boost::add_edge(index_to_vertex[linear_index],
                            index_to_vertex[below_linear_index],
                            graph);
        }
    }
}

/**
 * Generate bounding boxes from the connected components
 * 
 * The bounding boxes are generated by iterating over the labels and pixel
 * coordinates and updating the bounding box for each label.
 */
void ConnectedComponents::generate_boxes(const ushort width,
                                         const ushort height,
                                         const uint32_t min_spot_size) {
    auto labels = std::vector<int>(boost::num_vertices(graph));
    auto num_labels = boost::connected_components(graph, labels.data());

    // Initialize bounding boxes
    boxes = std::vector<Reflection>(num_labels, {width, height, 0, 0});

    // Iterate over the signals and update the bounding boxes
    for (const auto &[linear_index, signal] : signals) {
        // Retrieve the vertex index for this linear_index (linear_index -> vertex_id in build_graph)
        auto vertex_it = index_to_vertex.find(linear_index);

        if (vertex_it == index_to_vertex.end()) {
            throw std::runtime_error(
              fmt::format("Vertex ID not found for linear index {}", linear_index));
        }

        size_t vertex_id = vertex_it->second;  // Vertex ID in the graph

        int label = labels[vertex_id];  // Label assigned to this vertex

        auto &box = boxes[label];
        box.l = std::min(box.l, signal.x);
        box.r = std::max(box.r, signal.x);
        box.t = std::min(box.t, signal.y);
        box.b = std::max(box.b, signal.y);
        ++box.num_pixels;  // Increment the number of pixels in the box
    }

    uint num_unfiltered_spots = boxes.size();
    logger->info("Extracted {} spots", num_unfiltered_spots);

    // Filter boxes based on the minimum spot size
    if (min_spot_size > 0) {
        std::vector<Reflection> filtered_boxes;
        for (auto &box : boxes) {
            if (box.num_pixels >= min_spot_size) {
                filtered_boxes.emplace_back(box);
                num_strong_pixels_filtered += box.num_pixels;
            }
        }
        // Overwrite boxes with filtered boxes
        boxes = std::move(filtered_boxes);

        logger->info("Removed {} spots with size < {} pixels",
                     num_unfiltered_spots - boxes.size(),
                     min_spot_size);
    } else {
        num_strong_pixels_filtered = num_strong_pixels;
    }
}
#pragma endregion Connected Components

#pragma region 3D Connected Components
std::vector<Reflection3D> ConnectedComponents::find_3d_components(
  const std::vector<std::unique_ptr<ConnectedComponents>> &slices,
  const ushort width,
  const ushort height,
  const uint min_spot_size,
  const uint max_peak_centroid_separation) {
    /*
     * Initialize global containers for the 3D connected components
     */
    logger->debug("Initializing 3D connected components");
    // Graph for the 3D connected components
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_3d;
    // List to store each slice's mapping of linear_index -> global_vertex_id
    std::vector<std::unordered_map<size_t, size_t>> local_to_global_vertex_maps;
    // Global vertex ID counter starts at 0 and increments for each new vertex
    size_t global_vertex_id = 0;

    /*
     * Start building the 3D graph. First we copy the precomputed vertices and edges
     * from each slice's graph into the global 3D graph.
     */
    logger->debug("Building 3D graph");
    for (const auto &slice : slices) {
        // Get the slice's graph and vertex map
        const auto &graph = slice->get_graph();
        const auto &signals = slice->get_signals();

        // 2D linear_index -> global_vertex_id map for this slice
        std::unordered_map<size_t, size_t> local_to_global;

        // Add each vertex from the slice's graph to the global graph
        for (const auto &[linear_index, signal] : signals) {
            // Add the vertex to the global graph
            local_to_global[linear_index] = global_vertex_id++;
            boost::add_vertex(graph_3d);
        }

        // Move the local_to_global map to the global list
        local_to_global_vertex_maps.push_back(std::move(local_to_global));
    }

    /*
     * We then copy the pre-computed edges from each slice's graph to the 3D graph.
     */
    logger->debug("Copying edges to 3D graph");
    // Iterate over each slice and copy the edges to the 3D graph
    for (int i = 0; i < slices.size(); ++i) {
        // Current slice's 2d graph
        const auto &graph_2d = slices[i]->get_graph();
        // Current slice's vertex id -> linear index map
        const auto &vertex_to_index = slices[i]->get_vertex_to_index();

        logger->trace("Copying edges from slice {}", i);
        // Iterate over the edges in the slice's graph
        for (const auto &edge : boost::make_iterator_range(boost::edges(graph_2d))) {
            // Get the source and target vertices for the edge
            auto source_vertex = boost::source(edge, graph_2d);
            auto target_vertex = boost::target(edge, graph_2d);

            // Retrieve the original linear indices
            size_t source_linear_index = vertex_to_index.at(source_vertex);
            size_t target_linear_index = vertex_to_index.at(target_vertex);

            // Get the global vertex IDs
            size_t source_global_id =
              local_to_global_vertex_maps[i][source_linear_index];
            size_t target_global_id =
              local_to_global_vertex_maps[i][target_linear_index];

            // Add the edge to the 3D graph
            boost::add_edge(source_global_id, target_global_id, graph_3d);
        }
    }

    /*
     * Next, we add inter-slice connectivity to the 3D graph. This is done by
     * iterating over the local_to_global_vertex_maps and connecting vertices
     * that correspond to the same pixel in adjacent slices.
     */
    logger->debug("Adding inter-slice connectivity");
    // Loop through all slices except the last one
    for (size_t i = 0; i < slices.size() - 1; ++i) {
        const auto &current_vertex_map =
          local_to_global_vertex_maps[i];  // Current slice
        const auto &next_vertex_map = local_to_global_vertex_maps[i + 1];  // Next slice

        // Iterate over the vertices in the current slice
        for (const auto &[current_linear_index, current_global_id] :
             current_vertex_map) {
            // Check if the corresponding vertex exists in the next slice
            auto iterated_vertex = next_vertex_map.find(current_linear_index);
            // If it exists, connect the vertices in the 3D graph
            if (iterated_vertex != next_vertex_map.end()) {
                // Connect the vertices in the 3D graph
                size_t next_global_id =
                  iterated_vertex->second;  // Get the global id from the vertex map
                boost::add_edge(current_global_id, next_global_id, graph_3d);
            }
        }
    }

    /*
     * Now that we have constructed the 3D graph, we can perform connected components
     * analysis to find the 3D connected components.
     */
    logger->debug("Performing 3D connected components analysis");
    /*
     * Label vector for connected components. This matches vertex ids
     * to connected component labels.
     */
    std::vector<int> labels(boost::num_vertices(graph_3d));

    // Perform connected components analysis on the 3D graph 📈
    uint num_labels = boost::connected_components(graph_3d, labels.data());

    /*
     * Group the 3D connected components by their labels and compute the bounding boxes
     * and weighted centers of mass for each component. We do this by creating a map
     * of labels to vectors of global vertex IDs, and then iterating over each map
     * entry to compute the bounding box and center of mass.
     */
    logger->debug("Grouping 3D connected components");

    // Map of labels -> list of global vertices for each label
    std::unordered_map<int, std::vector<size_t>> label_to_vertices;
    for (int i = 0; i < labels.size(); ++i) {
        // Add the vertex to the label's list
        label_to_vertices[labels[i]].push_back(i);
    }

    std::vector<Reflection3D> reflections_3d(num_labels);  // List of 3D reflections

    /*
     * Iterate through each slice. The index corresponds to the
     * z-coordinate in this 3-D stack of slices. Order of these
     * slices should have been preserved during the construction
     * of the slice list.
     */
    for (int z = 0; z < slices.size(); ++z) {
        // Current working slice
        const auto &slice = slices[z];
        // All signals in the slice
        auto &signals = slice->get_signals();  // Not const because we need to update z
        // Slice index to vertex mapping
        const auto &index_to_vertex = slice->get_index_to_vertex();

        /*
         * Iterate through each signal in the slice and update its 3D
         * reflection with the corresponding label from the 3D connected
         * components analysis.
         */
        for (auto &[linear_index, signal] : signals) {
            const auto &local_to_global_vertex_map = local_to_global_vertex_maps[z];
            // Retrieve the vertex ID for this linear index
            auto vertex_it = local_to_global_vertex_map.find(linear_index);

            if (vertex_it == local_to_global_vertex_map.end()) {
                throw std::runtime_error(
                  fmt::format("Vertex ID not found for linear index {}", linear_index));
            }

            size_t vertex_id = vertex_it->second;  // Vertex ID in the 3D graph

            // Get the label for this vertex
            int label = labels[vertex_id];

            // Get the reflection for this label
            auto &reflection = reflections_3d[label];

            // Calculate DIALS z-index by reversing the slice index
            auto z_index = slices.size() - z - 1;
            // Add z-index to the signal
            signal.z = std::make_optional(z_index);

            // Update the reflection with the signal
            reflection.add_signal(signal);
        }
    }

    /*
     * Finally, filter the reflections based on the minimum spot size and
     * maximum peak centroid separation. Then return the filtered reflections.
     */
    uint initial_spot_count = reflections_3d.size();
    logger->info(
      fmt::format("Calculated {} spots", styled(initial_spot_count, fmt_cyan)));

    if (min_spot_size > 0) {
        logger->debug("Filtering reflections by minimum spot size");
        reflections_3d.erase(std::remove_if(reflections_3d.begin(),
                                            reflections_3d.end(),
                                            [min_spot_size](const auto &reflection) {
                                                return reflection.num_pixels
                                                       < min_spot_size;
                                            }),
                             reflections_3d.end());
    }

    logger->info(
      fmt::format("Filtered {} spots with size < {} pixels",
                  styled(initial_spot_count - reflections_3d.size(), fmt_cyan),
                  styled(min_spot_size, fmt_cyan)));
    uint filtered_spot_count = reflections_3d.size();

    if (max_peak_centroid_separation > 0) {
        logger->debug("Filtering reflections by maximum peak centroid separation");
        reflections_3d.erase(
          std::remove_if(reflections_3d.begin(),
                         reflections_3d.end(),
                         [max_peak_centroid_separation](const auto &reflection) {
                             return reflection.peak_centroid_distance()
                                    > max_peak_centroid_separation;
                         }),
          reflections_3d.end());
    }

    logger->info(
      fmt::format("Filtered {} spots with peak-centroid distance > {}",
                  styled(filtered_spot_count - reflections_3d.size(), fmt_cyan),
                  styled(max_peak_centroid_separation, fmt_cyan)));

    return reflections_3d;
}
#pragma endregion 3D Connected Components