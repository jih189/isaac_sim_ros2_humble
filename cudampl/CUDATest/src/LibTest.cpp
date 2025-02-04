#include "TestGraph.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    CUDAMPLib::TestGraph g(3);

    // Define some configurations as vectors of floats.
    vector<float> q1 = {0.0f, 1.0f, 2.0f};
    vector<float> q2 = {1.0f, 2.0f, 3.0f};
    vector<float> q3 = {2.0f, 3.0f, 4.0f};
    vector<float> q4 = {3.0f, 4.0f, 5.0f};
    vector<float> q5 = {4.0f, 5.0f, 6.0f};
    vector<float> q6 = {5.0f, 6.0f, 7.0f};

    vector<vector<float>> configuration_set_1;
    configuration_set_1.push_back(q1);
    configuration_set_1.push_back(q2);
    configuration_set_1.push_back(q3);
    configuration_set_1.push_back(q4);

    // Add states to the graph.
    g.add_states(configuration_set_1);

    vector<vector<float>> configuration_set_2;
    configuration_set_2.push_back(q5);
    configuration_set_2.push_back(q6);

    g.add_states(configuration_set_2);

    // if(g.contain(q2))
    // {
    //     cout << "q2 is in the graph" << endl;
    // }
    // else
    // {
    //     cout << "q2 is not in the graph" << endl;
    // }

    // // Connect the states.
    g.connect(q1, q2, 1.0);
    g.connect(q2, q3, 1.0);
    g.connect(q3, q4, 1.0);
    g.connect(q4, q5, 1.0);
    g.connect(q5, q6, 1.0);
    // // // Also add a direct edge from q1 to q3 for an alternative path.
    // // g.connect(q1, q3);

    // // // Check direct connection.
    cout << "q1 and q2 are directly connected: "  << g.is_connect(q1, q2) << endl;
    cout << "q1 and q3 are directly connected: "  << g.is_connect(q1, q3) << endl;

    // Find the shortest path from q1 to q4 using Dijkstra's algorithm.
    auto path = g.find_path(q1, q6);
    if (!path.empty()) {
        cout << "Shortest path from q1 to q6:" << endl;
        for (const auto& state : path) {
            for (float value : state)
                cout << value << " ";
            cout << endl;
        }
    }

    return 0;
}