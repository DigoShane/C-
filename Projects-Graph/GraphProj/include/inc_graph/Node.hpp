//This is a header file to be included in the main.
//This will allow use to use fn defns here.
#pragma once

#include <functional> //Needed for std::function - a wrapper that can hold any callable (function, lambda, functor)
#include <string>
#include <vector>

namespace inc_graph{

//A node is either an input node or a computed node.
enum class NodeKind { Input, Computed };

//Node represents one vertex in a computation graph.
struct Node {
    int id = -1; //Used as key in unordered_map<int, Node>
    NodeKind kind = NodeKind::Input;// accepts val, (Computed-> computes val from parents)

    std::vector<int> parents;// nodes this depends on
    std::vector<int> children;// nodes that depend on this.

    double value = 0.0;// value.
    bool orig = false;//implies val changed.

    std::function<double(const std::vector<double>&)> compute_fn;

    std::string name;
};

} // namespace inc_graph
