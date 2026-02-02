// incremental_graph_sim.cpp
//
// Reads one or more scenarios from a text file (DSL), builds the graph,
// runs recompute, applies changes, and prints everything nicely.
//
// Build:
//   g++ -std=c++17 -O2 -Wall -Wextra incremental_graph_sim.cpp -o inc_graph
//
// Run:
//   ./inc_graph scenarios.txt
//
// If no arg is given, defaults to "scenarios.txt".

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// --------------------------- Utilities ---------------------------

static std::string join_ids(const std::vector<int>& ids) {
    std::string s;
    for (size_t i = 0; i < ids.size(); ++i) {
        s += std::to_string(ids[i]);
        if (i + 1 < ids.size()) s += ", ";
    }
    return s;
}

static std::string join_set_sorted(const std::unordered_set<int>& s) {
    std::vector<int> v(s.begin(), s.end());
    std::sort(v.begin(), v.end());
    return join_ids(v);
}

static std::string trim(const std::string& x) {
    size_t i = 0, j = x.size();
    while (i < j && std::isspace(static_cast<unsigned char>(x[i]))) ++i;
    while (j > i && std::isspace(static_cast<unsigned char>(x[j - 1]))) --j;
    return x.substr(i, j - i);
}

static bool starts_with(const std::string& s, const std::string& pfx) {
    return s.size() >= pfx.size() && s.compare(0, pfx.size(), pfx) == 0;
}

// --------------------------- Node Model ---------------------------

enum class NodeKind { Input, Computed };

struct Node {
    int id = -1;
    NodeKind kind = NodeKind::Input;

    std::vector<int> parents;
    std::vector<int> children;

    double value = 0.0;
    bool dirty = false;

    std::function<double(const std::vector<double>&)> compute_fn;

    std::string name;   // optional label
    std::string expr;   // human-readable "equation" string for printing
};

// --------------------------- Engine ---------------------------

class IncrementalEngine {
public:
    void add_input_node(int id, double initial_value, std::string name = "") {
        ensure_unique_id(id);
        Node n;
        n.id = id;
        n.kind = NodeKind::Input;
        n.value = initial_value;
        n.dirty = false;
        n.name = std::move(name);
        n.expr = "(input)";
        nodes_.emplace(id, std::move(n));
    }

    void add_computed_node(
        int id,
        std::function<double(const std::vector<double>&)> compute_fn,
        std::vector<int> parents,
        std::string name = "",
        std::string expr = ""
    ) {
        ensure_unique_id(id);
        if (!compute_fn) throw std::invalid_argument("compute_fn must be non-empty");

        Node n;
        n.id = id;
        n.kind = NodeKind::Computed;
        n.compute_fn = std::move(compute_fn);
        n.parents = std::move(parents);
        n.value = 0.0;
        n.dirty = true;
        n.name = std::move(name);
        n.expr = std::move(expr);
        nodes_.emplace(id, std::move(n));
    }

    // Add dependency edge: parent -> child
    void add_edge(int parent, int child) {
        Node& p = get_node_ref(parent);
        Node& c = get_node_ref(child);
        p.children.push_back(child);
        c.parents.push_back(parent);
    }

    // Validate DAG (Kahn)
    void validate_acyclic() const {
        std::unordered_map<int, int> indeg;
        indeg.reserve(nodes_.size());

        for (const auto& kv : nodes_) indeg[kv.first] = 0;

        for (const auto& kv : nodes_) {
            const Node& n = kv.second;
            for (int ch : n.children) {
                auto it = indeg.find(ch);
                if (it == indeg.end()) {
                    throw std::runtime_error("Graph references unknown node id: " + std::to_string(ch));
                }
                it->second += 1;
            }
        }

        std::queue<int> q;
        for (const auto& kv : indeg) {
            if (kv.second == 0) q.push(kv.first);
        }

        int popped = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            ++popped;
            const Node& nu = nodes_.at(u);
            for (int v : nu.children) {
                indeg[v]--;
                if (indeg[v] == 0) q.push(v);
            }
        }

        if (popped != static_cast<int>(nodes_.size())) {
            throw std::runtime_error("Cycle detected: graph is not acyclic.");
        }
    }

    void set_input(int id, double new_value, bool trace = false) {
        Node& n = get_node_ref(id);
        if (n.kind != NodeKind::Input) {
            throw std::runtime_error("set_input on non-input node id=" + std::to_string(id));
        }

        if (trace) {
            std::cout << "\n  [set_input] " << label(n) << " = " << new_value << "\n";
        }

        if (n.value == new_value) {
            if (trace) std::cout << "    value unchanged; no invalidation.\n";
            return;
        }

        n.value = new_value;
        mark_dirty_and_propagate(id, trace);
    }

    void recompute(bool trace = false) {
        std::unordered_set<int> dirty_nodes;
        for (const auto& kv : nodes_) {
            if (kv.second.dirty) dirty_nodes.insert(kv.first);
        }

        if (dirty_nodes.empty()) {
            if (trace) std::cout << "\n  [recompute] No dirty nodes.\n";
            return;
        }

        std::unordered_map<int, int> dirty_indeg;
        dirty_indeg.reserve(dirty_nodes.size());
        for (int id : dirty_nodes) dirty_indeg[id] = 0;

        for (int id : dirty_nodes) {
            const Node& n = nodes_.at(id);
            for (int p : n.parents) {
                if (dirty_nodes.count(p)) dirty_indeg[id] += 1;
            }
        }

        std::queue<int> q;
        for (const auto& kv : dirty_indeg) {
            if (kv.second == 0) q.push(kv.first);
        }

        if (trace) {
            std::cout << "\n  [recompute] Dirty nodes: {" << join_set_sorted(dirty_nodes) << "}\n";
        }

        int processed = 0;
        while (!q.empty()) {
            int id = q.front(); q.pop();
            ++processed;

            Node& n = get_node_ref(id);

            if (n.kind == NodeKind::Input) {
                if (trace) std::cout << "    clean input  " << label(n) << " (value=" << n.value << ")\n";
                n.dirty = false;
            } else {
                std::vector<double> parent_vals;
                parent_vals.reserve(n.parents.size());
                for (int p : n.parents) {
                    const Node& pn = nodes_.at(p);
                    if (pn.dirty) {
                        throw std::runtime_error("Internal error: parent still dirty for node id=" + std::to_string(id));
                    }
                    parent_vals.push_back(pn.value);
                }

                double old = n.value;
                double newv = n.compute_fn(parent_vals);
                n.value = newv;
                n.dirty = false;

                if (trace) {
                    std::cout << "    compute      " << label(n)
                              << ": " << old << " -> " << newv
                              << " (parents=[" << join_ids(n.parents) << "])\n";
                }
            }

            for (int ch : n.children) {
                if (!dirty_nodes.count(ch)) continue;
                dirty_indeg[ch] -= 1;
                if (dirty_indeg[ch] == 0) q.push(ch);
            }
        }

        if (processed != static_cast<int>(dirty_nodes.size())) {
            throw std::runtime_error("Could not process all dirty nodes (cycle/bug).");
        }
    }

    double get_value(int id) const {
        const Node& n = nodes_.at(id);
        if (n.dirty) {
            throw std::runtime_error("Value requested for dirty node id=" + std::to_string(id));
        }
        return n.value;
    }

    void print_summary(const std::string& title) const {
        std::cout << "\n============================================================\n";
        std::cout << title << "\n";
        std::cout << "============================================================\n";

        std::vector<int> ids;
        ids.reserve(nodes_.size());
        for (const auto& kv : nodes_) ids.push_back(kv.first);
        std::sort(ids.begin(), ids.end());

        std::cout << "\n[NODES]\n";
        for (int id : ids) {
            const Node& n = nodes_.at(id);
            std::cout << "  - " << label(n)
                      << "  kind=" << (n.kind == NodeKind::Input ? "Input" : "Computed")
                      << "  expr=" << n.expr
                      << "\n";
            std::cout << "      parents=[" << join_ids(n.parents) << "]  children=[" << join_ids(n.children) << "]\n";
        }

        std::cout << "\n[VALUES]\n";
        for (int id : ids) {
            const Node& n = nodes_.at(id);
            std::cout << "  - " << label(n)
                      << "  value=" << n.value
                      << "  dirty=" << (n.dirty ? "true" : "false")
                      << "\n";
        }
    }

    // after all computed nodes added with parents list, connect edges accordingly
    void materialize_edges_from_parents() {
        // Clear children first (in case reused, though we create new engine per scenario)
        for (auto& kv : nodes_) kv.second.children.clear();

        // For every node, add edges p -> node.id
        for (auto& kv : nodes_) {
            Node& n = kv.second;
            for (int p : n.parents) {
                Node& pn = get_node_ref(p);
                pn.children.push_back(n.id);
            }
        }
    }

private:
    std::unordered_map<int, Node> nodes_;

    void ensure_unique_id(int id) const {
        if (nodes_.count(id)) {
            throw std::runtime_error("Duplicate node id: " + std::to_string(id));
        }
    }

    Node& get_node_ref(int id) {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) throw std::runtime_error("Unknown node id: " + std::to_string(id));
        return it->second;
    }

    std::string label(const Node& n) const {
        if (!n.name.empty()) return n.name + "(" + std::to_string(n.id) + ")";
        return "node(" + std::to_string(n.id) + ")";
    }

    void mark_dirty_and_propagate(int start_id, bool trace) {
        std::queue<int> q;
        std::unordered_set<int> seen;
        q.push(start_id);
        seen.insert(start_id);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            Node& nu = get_node_ref(u);

            if (!nu.dirty) {
                nu.dirty = true;
                if (trace) std::cout << "    mark dirty: " << label(nu) << "\n";
            } else {
                if (trace) std::cout << "    already dirty: " << label(nu) << "\n";
            }

            for (int ch : nu.children) {
                if (!seen.count(ch)) {
                    seen.insert(ch);
                    q.push(ch);
                }
            }
        }
    }
};

// --------------------------- Scenario Model + Parser ---------------------------

struct Change {
    int id = -1;
    double new_value = 0.0;
};

struct ComputedSpec {
    int id = -1;
    std::string name;
    std::string op;
    std::vector<int> args_ids;     // parent IDs
    double param = 0.0;            // for scale/add_const
    bool has_param = false;
    std::string expr;
};

struct Scenario {
    std::string name;
    std::vector<std::tuple<int,std::string,double>> inputs; // (id,name,value)
    std::vector<ComputedSpec> computed;
    std::vector<Change> changes;
    std::vector<std::pair<int,int>> extra_edges; // e.g. cycle_edge
};

static std::vector<std::string> tokenize(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> t;
    std::string w;
    while (iss >> w) t.push_back(w);
    return t;
}

static double to_double(const std::string& s) {
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) throw std::runtime_error("Bad number: " + s);
    return v;
}

static int to_int(const std::string& s) {
    size_t pos = 0;
    int v = std::stoi(s, &pos);
    if (pos != s.size()) throw std::runtime_error("Bad int: " + s);
    return v;
}

static std::vector<Scenario> load_scenarios(const std::string& path) {
    std::ifstream fp(path);
    if (!fp) throw std::runtime_error("Could not open scenario file: " + path);

    std::vector<Scenario> scenarios;
    Scenario cur;
    bool in = false;

    std::string line;
    int lineno = 0;
    while (std::getline(fp, line)) {
        ++lineno;
        line = trim(line);
        if (line.empty() || starts_with(line, "#")) continue;

        auto tok = tokenize(line);
        if (tok.empty()) continue;

        if (tok[0] == "scenario") {
            if (tok.size() < 2) throw std::runtime_error("Line " + std::to_string(lineno) + ": scenario needs a name");
            if (in) throw std::runtime_error("Line " + std::to_string(lineno) + ": nested scenario not allowed");
            in = true;
            cur = Scenario{};
            cur.name = tok[1];
        }
        else if (tok[0] == "end") {
            if (!in) throw std::runtime_error("Line " + std::to_string(lineno) + ": 'end' without scenario");
            scenarios.push_back(cur);
            in = false;
        }
        else if (!in) {
            throw std::runtime_error("Line " + std::to_string(lineno) + ": directive outside scenario: " + tok[0]);
        }
        else if (tok[0] == "input") {
            if (tok.size() < 4) throw std::runtime_error("Line " + std::to_string(lineno) + ": input <id> <name> <value>");
            int id = to_int(tok[1]);
            std::string name = tok[2];
            double val = to_double(tok[3]);
            cur.inputs.push_back({id, name, val});
        }
        else if (tok[0] == "computed") {
            // computed <id> <name> <op> <args...>
            if (tok.size() < 4) throw std::runtime_error("Line " + std::to_string(lineno) + ": computed <id> <name> <op> ...");
            ComputedSpec cs;
            cs.id = to_int(tok[1]);
            cs.name = tok[2];
            cs.op = tok[3];

            // ops:
            // add a b
            // sub a b
            // mul a b
            // scale a k
            // add_const a k
            // sum a b c ...
            if (cs.op == "add" || cs.op == "sub" || cs.op == "mul") {
                if (tok.size() != 6) throw std::runtime_error("Line " + std::to_string(lineno) + ": " + cs.op + " needs 2 ids");
                cs.args_ids = {to_int(tok[4]), to_int(tok[5])};
                cs.expr = cs.name + " = " + cs.op + "(" + tok[4] + "," + tok[5] + ")";
            } else if (cs.op == "scale" || cs.op == "add_const") {
                if (tok.size() != 6) throw std::runtime_error("Line " + std::to_string(lineno) + ": " + cs.op + " needs 1 id and 1 param");
                cs.args_ids = {to_int(tok[4])};
                cs.param = to_double(tok[5]);
                cs.has_param = true;
                cs.expr = cs.name + " = " + cs.op + "(" + tok[4] + "," + tok[5] + ")";
            } else if (cs.op == "sum") {
                if (tok.size() < 5) throw std::runtime_error("Line " + std::to_string(lineno) + ": sum needs >= 1 id");
                for (size_t i = 4; i < tok.size(); ++i) cs.args_ids.push_back(to_int(tok[i]));
                cs.expr = cs.name + " = sum(" + join_ids(cs.args_ids) + ")";
            } else {
                throw std::runtime_error("Line " + std::to_string(lineno) + ": unknown op: " + cs.op);
            }

            cur.computed.push_back(std::move(cs));
        }
        else if (tok[0] == "change") {
            if (tok.size() != 3) throw std::runtime_error("Line " + std::to_string(lineno) + ": change <id> <new_value>");
            Change ch;
            ch.id = to_int(tok[1]);
            ch.new_value = to_double(tok[2]);
            cur.changes.push_back(ch);
        }
        else if (tok[0] == "cycle_edge") {
            if (tok.size() != 3) throw std::runtime_error("Line " + std::to_string(lineno) + ": cycle_edge <u> <v>");
            int u = to_int(tok[1]);
            int v = to_int(tok[2]);
            cur.extra_edges.push_back({u,v});
        }
        else {
            throw std::runtime_error("Line " + std::to_string(lineno) + ": unknown directive: " + tok[0]);
        }
    }

    if (in) throw std::runtime_error("Scenario file ended without 'end'");

    return scenarios;
}

static std::function<double(const std::vector<double>&)>
make_compute_fn(const ComputedSpec& cs) {
    if (cs.op == "add") {
        return [](const std::vector<double>& p){ return p.at(0) + p.at(1); };
    }
    if (cs.op == "sub") {
        return [](const std::vector<double>& p){ return p.at(0) - p.at(1); };
    }
    if (cs.op == "mul") {
        return [](const std::vector<double>& p){ return p.at(0) * p.at(1); };
    }
    if (cs.op == "scale") {
        double k = cs.param;
        return [k](const std::vector<double>& p){ return k * p.at(0); };
    }
    if (cs.op == "add_const") {
        double k = cs.param;
        return [k](const std::vector<double>& p){ return p.at(0) + k; };
    }
    if (cs.op == "sum") {
        return [](const std::vector<double>& p){
            double s = 0.0;
            for (double x : p) s += x;
            return s;
        };
    }
    throw std::runtime_error("Internal: unsupported op " + cs.op);
}

// --------------------------- Runner ---------------------------

static void run_scenario(const Scenario& sc, bool trace) {
    IncrementalEngine eng;

    // Build nodes
    for (const auto& in : sc.inputs) {
        int id; std::string name; double v;
        std::tie(id, name, v) = in;
        eng.add_input_node(id, v, name);
    }
    for (const auto& cs : sc.computed) {
        auto fn = make_compute_fn(cs);
        eng.add_computed_node(cs.id, fn, cs.args_ids, cs.name, cs.expr);
    }

    // Materialize edges based on parents lists
    eng.materialize_edges_from_parents();

    // Extra edges (e.g. explicit cycle)
    for (const auto& e : sc.extra_edges) {
        eng.add_edge(e.first, e.second);
    }

    // Print scenario header
    eng.print_summary("SCENARIO: " + sc.name + "  (before validate/recompute)");

    // Validate DAG (may throw)
    std::cout << "\n[validate_acyclic] ... ";
    eng.validate_acyclic();
    std::cout << "OK\n";

    // Initial compute
    std::cout << "\n[initial recompute]\n";
    eng.recompute(trace);
    eng.print_summary("SCENARIO: " + sc.name + "  (after initial recompute)");

    // Apply changes
    for (size_t i = 0; i < sc.changes.size(); ++i) {
        const auto& ch = sc.changes[i];
        std::cout << "\n------------------------------------------------------------\n";
        std::cout << "CHANGE " << (i + 1) << "/" << sc.changes.size()
                  << ": set_input(id=" << ch.id << ", value=" << ch.new_value << ")\n";
        std::cout << "------------------------------------------------------------\n";
        eng.set_input(ch.id, ch.new_value, trace);
        eng.recompute(trace);
        eng.print_summary("SCENARIO: " + sc.name + "  (after change " + std::to_string(i + 1) + ")");
    }
}

int main(int argc, char** argv) {
    try {
        std::string path = (argc >= 2) ? argv[1] : "scenarios.txt";
        auto scenarios = load_scenarios(path);

        std::cout << "============================================================\n";
        std::cout << "Incremental Dependency Graph Simulator\n";
        std::cout << "Reading scenarios from: " << path << "\n";
        std::cout << "============================================================\n";

        bool trace = true; // set false if you want quieter output

        for (size_t i = 0; i < scenarios.size(); ++i) {
            std::cout << "\n\n############################################################\n";
            std::cout << "RUN " << (i + 1) << "/" << scenarios.size() << "\n";
            std::cout << "############################################################\n";
            run_scenario(scenarios[i], trace);
        }

        std::cout << "\n\nAll scenarios completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}

