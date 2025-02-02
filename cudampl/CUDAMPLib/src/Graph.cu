#include "Graph.h"
#include <algorithm>   // for std::reverse
#include <limits>
#include <cmath>
#include <stdexcept>

namespace CUDAMPLib{

    __global__ void compute_distance_kernel(float* d_q, float* d_graph, float* d_distance_map, int num_of_added_states, int dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_added_states)
            return;

        float sum = 0.0;
        for (int i = 0; i < dim; i++) {
            float diff = d_q[i] - d_graph[idx * dim + i];
            sum += diff * diff;
        }

        d_distance_map[idx] = sqrt(sum);
    }

    std::vector<float> Graph::compute_distance(const std::vector<float> &q) const
    {
        if(dim != q.size()){
            throw std::runtime_error("Configuration dimension does not match the graph dimension.");
        }

        // prepare device memory for q
        float* d_q;
        int d_q_bytes = dim * sizeof(float);

        cudaMalloc(&d_q, d_q_bytes);

        // copy q to d_q
        cudaMemcpy(d_q, q.data(), d_q_bytes, cudaMemcpyHostToDevice);

        int size_of_graph = boost::num_vertices(g);

        // use kernel to compute distance
        compute_distance_kernel<<<(size_of_graph + 255) / 256, 256>>>(d_q, d_graph, d_distance_map, size_of_graph, dim);

        // wait for kernel to finish
        cudaDeviceSynchronize();

        // copy d_distance_map to host memory
        std::vector<float> distance_map(size_of_graph);
        cudaMemcpy(distance_map.data(), d_distance_map, size_of_graph * sizeof(float), cudaMemcpyDeviceToHost);

        // deallocate device memory
        cudaFree(d_q);

        return distance_map;
    }

    // Helper: find a vertex by comparing its stored configuration.
    Graph::BoostVertex Graph::find_vertex_by_config(const std::vector<float>& q) const {
        
        if (dim != q.size()) {
            throw std::runtime_error("Configuration dimension does not match the graph dimension.");
        }

        // compute distance between q and all vertices
        std::vector<float> distance_map = compute_distance(q);

        // find the vertex with distance 0
        for (size_t i = 0; i < distance_map.size(); i++) {
            if (distance_map[i] == 0) {
                return boost::vertex(i, g);
            }
        }
        return boost::graph_traits<BoostGraph>::null_vertex();
    }

    void Graph::add_state(const std::vector<float>& q) {
        BoostVertex v = boost::add_vertex(g);
        g[v].configuration = q;
    }

    void Graph::add_states(const std::vector<std::vector<float>>& qs) {

        if (qs.empty()) {
            return;
        }

        // ensure qs has no duplicate configurations
        std::vector<std::vector<float>> qs_unique = qs;
        std::sort(qs_unique.begin(), qs_unique.end());
        qs_unique.erase(std::unique(qs_unique.begin(), qs_unique.end()), qs_unique.end());

        int num_of_added_states = qs_unique.size();

        // get the size of the graph
        int size_of_graph = boost::num_vertices(g);

        if (size_of_graph > 0)
        {
            int d_new_graph_bytes = (size_of_graph + num_of_added_states) * dim * sizeof(float);
            int d_new_distance_map_bytes = (size_of_graph + num_of_added_states) * sizeof(float);

            float* d_new_graph;
            float* d_new_distance_map;

            cudaMalloc(&d_new_graph, d_new_graph_bytes);
            cudaMalloc(&d_new_distance_map, d_new_distance_map_bytes);

            // copy old graph to new graph in device memory
            cudaMemcpy(d_new_graph, d_graph, size_of_graph * dim * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_new_distance_map, d_distance_map, size_of_graph * sizeof(float), cudaMemcpyDeviceToDevice);

            // copy new qs to new graph in device memory
            std::vector<float> qs_flatten = floatVectorFlatten(qs_unique);
            cudaMemcpy(d_new_graph + size_of_graph * dim, qs_flatten.data(), num_of_added_states * dim * sizeof(float), cudaMemcpyHostToDevice);

            // deallocate old graph
            cudaFree(d_graph);
            cudaFree(d_distance_map);

            // set d_graph to new graph
            d_graph = d_new_graph;
            d_distance_map = d_new_distance_map;
        }
        else{
            // prepare d_graph with size of num_of_added_states * dim in gpu memory
            int d_graph_bytes = num_of_added_states * dim * sizeof(float);
            int d_distance_map_bytes = num_of_added_states * sizeof(float);

            cudaMalloc(&d_graph, d_graph_bytes);
            cudaMalloc(&d_distance_map, d_distance_map_bytes);

            // set has_d_graph to true
            has_d_graph = true;

            // copy qs to d_graph
            std::vector<float> qs_flatten = floatVectorFlatten(qs_unique);

            cudaMemcpy(d_graph, qs_flatten.data(), d_graph_bytes, cudaMemcpyHostToDevice);
        }

        // add states to the graph
        for (const auto& q : qs_unique) {
            add_state(q);
        }
    }

    // deconstructor
    Graph::~Graph() {
        if (has_d_graph) {
            cudaFree(d_graph);
            cudaFree(d_distance_map);
        }
    }

    bool Graph::contain(const std::vector<float>& q) const{
        // check the dimension of the configuration
        if (q.size() != dim) {
            throw std::runtime_error("Configuration dimension does not match the graph dimension.");
        }
        // call compute_distance to get distance_map
        std::vector<float> distance_map = compute_distance(q);
        // check if distance_map contains 0
        return std::find(distance_map.begin(), distance_map.end(), 0) != distance_map.end();
    }

    void Graph::connect(const std::vector<float>& q1, const std::vector<float>& q2, float weight) {
        BoostVertex v1 = find_vertex_by_config(q1);
        BoostVertex v2 = find_vertex_by_config(q2);
        if (v1 == boost::graph_traits<BoostGraph>::null_vertex() ||
            v2 == boost::graph_traits<BoostGraph>::null_vertex())
        {
            std::cerr << "Error: One of the states does not exist!" << std::endl;
            return;
        }
        // Avoid duplicate edges.
        if (!boost::edge(v1, v2, g).second) {
            boost::add_edge(v1, v2, Graph::EdgeProperties{weight}, g);
        }
    }

    bool Graph::is_connect(const std::vector<float>& q1, const std::vector<float>& q2) const {
        BoostVertex v1 = find_vertex_by_config(q1);
        BoostVertex v2 = find_vertex_by_config(q2);
        if (v1 == boost::graph_traits<BoostGraph>::null_vertex() ||
            v2 == boost::graph_traits<BoostGraph>::null_vertex())
        {
            return false;
        }
        return boost::edge(v1, v2, g).second;
    }

    std::vector<std::vector<float>> Graph::find_path(const std::vector<float>& q1, const std::vector<float>& q2) const {
        std::vector<std::vector<float>> path;
        BoostVertex source = find_vertex_by_config(q1);
        BoostVertex target = find_vertex_by_config(q2);
        if (source == boost::graph_traits<BoostGraph>::null_vertex() ||
            target == boost::graph_traits<BoostGraph>::null_vertex())
        {
            std::cerr << "Error: One of the states does not exist!" << std::endl;
            return path;
        }

        const auto num_vertices = boost::num_vertices(g);
        std::vector<double> distances(num_vertices, std::numeric_limits<double>::max());
        std::vector<BoostVertex> predecessors(num_vertices, boost::graph_traits<BoostGraph>::null_vertex());

        auto indexMap = boost::get(boost::vertex_index, g);
        // Get the weight map explicitly from the bundled edge property.
        auto weight_map = boost::get(&EdgeProperties::weight, g);

        // Run Dijkstra's algorithm from the source.
        boost::dijkstra_shortest_paths(g, source,
            boost::weight_map(weight_map)
            .distance_map(boost::make_iterator_property_map(distances.begin(), indexMap))
            .predecessor_map(boost::make_iterator_property_map(predecessors.begin(), indexMap))
        );

        // Check if target is reachable.
        if (distances[indexMap[target]] == std::numeric_limits<double>::max()){
            std::cerr << "No path found between the given states." << std::endl;
            return path;
        }

        // Reconstruct the path from target back to source using the predecessor map.
        std::vector<BoostVertex> vertexPath;
        for (BoostVertex v = target; v != source; v = predecessors[indexMap[v]]) {
            if (predecessors[indexMap[v]] == boost::graph_traits<BoostGraph>::null_vertex()){
                std::cerr << "No path found between the given states." << std::endl;
                return std::vector<std::vector<float>>();
            }
            vertexPath.push_back(v);
        }
        vertexPath.push_back(source);
        std::reverse(vertexPath.begin(), vertexPath.end());

        // Convert the vertex path to a vector of configurations.
        for (auto v : vertexPath) {
            path.push_back(g[v].configuration);
        }
        return path;
    }
    
} // namespace CUDAMPLib