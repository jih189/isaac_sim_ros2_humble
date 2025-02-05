#pragma once

#include <base/States.h>
#include <base/Motions.h>

namespace CUDAMPLib
{
    /**
        This is used for get motion to k nearest neighbors.
        index_in_states: the index of the state in the states.
        index_in_graph: the index of the state in the graph.
     */
    struct StateIndexPair{
        size_t index_in_states;
        size_t index_in_graph;
    };

    class BaseGraph {
        public:
            BaseGraph() {}

            virtual ~BaseGraph() {}

            // Add a states to the graph.
            virtual void add_states(const BaseStatesPtr & states) = 0;

            // Add start states to the graph.
            virtual void add_start_states(const BaseStatesPtr & states) = 0;

            // Add goal states to the graph.
            virtual void add_goal_states(const BaseStatesPtr & states) = 0;

            // Get motions to k nearest neighbors.
            virtual BaseMotionsPtr get_motions_to_k_nearest_neighbors(const BaseStatesPtr & states, int k, std::vector<StateIndexPair> & pairs) = 0;

            // Returns true if the configuration q is in the graph.
            virtual void contain(const BaseStatesPtr & states, int * d_result) const = 0;

            // Connect two states by adding an edge between them.
            // The edge weight is the Euclidean distance between the configurations.
            virtual void connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, float * d_weight) = 0;

            // Check if two states are directly connected by an edge.
            virtual void is_connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, int * d_result) const = 0;
    
            // debug print
            virtual void print() const = 0;
    };

    typedef std::shared_ptr<BaseGraph> BaseGraphPtr;

} // namespace CUDAMPLib