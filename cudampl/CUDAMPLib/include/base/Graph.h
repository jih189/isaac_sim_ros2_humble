#pragma once

#include <base/States.h>

namespace CUDAMPLib
{
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