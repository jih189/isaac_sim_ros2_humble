#include "Graph.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    CUDAMPLib::Graph g;

    // Define some configurations as vectors of floats.
    vector<float> q1 = {0.0f, 1.0f, 2.0f};
    vector<float> q2 = {1.0f, 2.0f, 3.0f};
    vector<float> q3 = {2.0f, 3.0f, 4.0f};
    vector<float> q4 = {3.0f, 4.0f, 5.0f};

    // Add states to the graph.
    g.add_state(q1);
    g.add_state(q2);
    g.add_state(q3);
    g.add_state(q4);

    // Connect the states.
    g.connect(q1, q2);
    g.connect(q2, q3);
    g.connect(q3, q4);
    // Also add a direct edge from q1 to q3 for an alternative path.
    g.connect(q1, q3);

    // Check direct connection.
    cout << "q1 and q2 are directly connected: " << boolalpha << g.is_connect(q1, q2) << endl;
    cout << "q1 and q3 are directly connected: " << boolalpha << g.is_connect(q1, q3) << endl;

    // Find the shortest path from q1 to q4 using Dijkstra's algorithm.
    auto path = g.find_path(q1, q4);
    if (!path.empty()) {
        cout << "Shortest path from q1 to q4:" << endl;
        for (const auto& state : path) {
            for (float value : state)
                cout << value << " ";
            cout << endl;
        }
    }

    return 0;
}