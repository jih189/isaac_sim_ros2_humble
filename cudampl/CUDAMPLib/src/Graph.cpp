#include "Graph.h"
#include <algorithm>   // for std::reverse
#include <limits>
#include <cmath>
#include <stdexcept>

namespace CUDAMPLib{

    // Compute the Euclidean distance between two configurations.
    double euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
        if(a.size() != b.size()){
            throw std::runtime_error("Configurations must have the same dimension.");
        }
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    //
    // Implementation of Graph member functions
    //

    // Helper: find a vertex by comparing its stored configuration.
    Graph::Vertex Graph::find_vertex_by_config(const std::vector<float>& q) const {
        auto verticesPair = boost::vertices(g);
        for (auto it = verticesPair.first; it != verticesPair.second; ++it) {
            if (g[*it].configuration == q)
                return *it;
        }
        return boost::graph_traits<BoostGraph>::null_vertex();
    }

    void Graph::add_state(const std::vector<float>& q) {
        if (contain(q))
            return;
        Vertex v = boost::add_vertex(g);
        g[v].configuration = q;
    }

    bool Graph::contain(const std::vector<float>& q) const {
        return find_vertex_by_config(q) != boost::graph_traits<BoostGraph>::null_vertex();
    }

    void Graph::connect(const std::vector<float>& q1, const std::vector<float>& q2) {
        Vertex v1 = find_vertex_by_config(q1);
        Vertex v2 = find_vertex_by_config(q2);
        if (v1 == boost::graph_traits<BoostGraph>::null_vertex() ||
            v2 == boost::graph_traits<BoostGraph>::null_vertex())
        {
            std::cerr << "Error: One of the states does not exist!" << std::endl;
            return;
        }
        // Avoid duplicate edges.
        if (!boost::edge(v1, v2, g).second) {
            double weight = euclidean_distance(q1, q2);
            boost::add_edge(v1, v2, Graph::EdgeProperties{weight}, g);
        }
    }

    bool Graph::is_connect(const std::vector<float>& q1, const std::vector<float>& q2) const {
        Vertex v1 = find_vertex_by_config(q1);
        Vertex v2 = find_vertex_by_config(q2);
        if (v1 == boost::graph_traits<BoostGraph>::null_vertex() ||
            v2 == boost::graph_traits<BoostGraph>::null_vertex())
        {
            return false;
        }
        return boost::edge(v1, v2, g).second;
    }

    std::vector<std::vector<float>> Graph::find_path(const std::vector<float>& q1, const std::vector<float>& q2) const {
        std::vector<std::vector<float>> path;
        Vertex source = find_vertex_by_config(q1);
        Vertex target = find_vertex_by_config(q2);
        if (source == boost::graph_traits<BoostGraph>::null_vertex() ||
            target == boost::graph_traits<BoostGraph>::null_vertex())
        {
            std::cerr << "Error: One of the states does not exist!" << std::endl;
            return path;
        }

        const auto num_vertices = boost::num_vertices(g);
        std::vector<double> distances(num_vertices, std::numeric_limits<double>::max());
        std::vector<Vertex> predecessors(num_vertices, boost::graph_traits<BoostGraph>::null_vertex());

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
        std::vector<Vertex> vertexPath;
        for (Vertex v = target; v != source; v = predecessors[indexMap[v]]) {
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