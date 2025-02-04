#pragma once

#include "base/Graph.h"

namespace CUDAMPLib
{
    class SingleArmGraph : public BaseGraph
    {
        public:
            SingleArmGraph(int dim);

            ~SingleArmGraph();

            // Add a states to the graph.
            void add_states(const BaseStatesPtr & states) override;

            // Returns true if the configuration q is in the graph.
            void contain(const BaseStatesPtr & states, int * d_result) const override;

            // Connect two states by adding an edge between them.
            // The edge weight is the Euclidean distance between the configurations.
            void connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, float * d_weight) override;

            // Check if two states are directly connected by an edge.
            void is_connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, int * d_result) const override;

        private:

            float * d_states_in_graph; // states in graph
    };
} // namespace CUDAMPLib