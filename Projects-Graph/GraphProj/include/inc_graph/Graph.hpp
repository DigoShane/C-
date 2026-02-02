#pragma once

#include "inc_graph/Node.hpp" //include Node & NodeKind.
#include <unordered_map> //hash table container used to store nodes by ID.
#include <string>

namespace inc_graph {

class Graph {
public:
    void add_input_node(int id, double initial_value, std::string name = "");
    void add_computed_node(int id,
                           std::function<double(const std::vector<double>&)> compute_fn,
                           std::string name = "");
    void add_edge(int parent, int child);

    void validate_acyclic() const; //Checks that the graph is a DAG (no directed cycles).

    Node& node(int id); //Returns a modifiable reference to the node with given id.
    const Node& node(int id) const;

    const std::unordered_map<int, Node>& nodes() const { return nodes_; }
    std::unordered_map<int, Node>& nodes() { return nodes_; }

private:
    std::unordered_map<int, Node> nodes_;
    void ensure_unique_id(int id) const;
};

} // namespace inc_graph

