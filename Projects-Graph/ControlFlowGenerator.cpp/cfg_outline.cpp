#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

// A simple node representing a control-flow block.
struct CFNode {
    std::string label;
    std::vector<CFNode*> children;
};

// Trim helpers (very basic)
static std::string ltrim(std::string s) {
    while (!s.empty() && std::isspace((unsigned char)s[0])) s.erase(s.begin());
    return s;
}

static std::string rtrim(std::string s) {
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    return s;
}

static std::string trim(std::string s) {
    return rtrim(ltrim(s));
}

// Checks if a line starts with a keyword (very naive)
static bool starts_with_kw(const std::string& line, const std::string& kw) {
    if (line.size() < kw.size()) return false;
    if (line.compare(0, kw.size(), kw) != 0) return false;
    // require next char boundary-ish
    if (line.size() == kw.size()) return true;
    char c = line[kw.size()];
    return std::isspace((unsigned char)c) || c == '(';
}

// Pretty tree printing (ASCII)
static void print_tree(const CFNode* node, const std::string& prefix = "", bool is_last = true) {
    std::cout << prefix;
    std::cout << (is_last ? "└── " : "├── ");
    std::cout << node->label << "\n";

    for (size_t i = 0; i < node->children.size(); i++) {
        bool last_child = (i + 1 == node->children.size());
        std::string new_prefix = prefix + (is_last ? "    " : "│   ");
        print_tree(node->children[i], new_prefix, last_child);
    }
}

// Very rough “extract label” for control statements
static std::string extract_control_label(const std::string& line) {
    // Keep the whole line but shorten if too long.
    std::string s = trim(line);
    if ((int)s.size() > 80) s = s.substr(0, 77) + "...";
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <cpp_file>\n";
        return 0;
    }

    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cerr << "Could not open file: " << argv[1] << "\n";
        return 1;
    }

    // Store nodes so pointers remain valid (novice approach: heap + manual cleanup)
    std::vector<CFNode*> all_nodes;

    // Root node representing the file
    CFNode* root = new CFNode();
    root->label = std::string("FILE: ") + argv[1];
    all_nodes.push_back(root);

    // Stack for nesting. The top is the current parent.
    std::vector<CFNode*> stack;
    stack.push_back(root);

    // Track braces to know when to pop
    int brace_depth = 0;
    std::vector<int> brace_depth_stack; // brace depth where each node started
    brace_depth_stack.push_back(0);

    std::string raw;
    int line_no = 0;

    while (std::getline(fin, raw)) {
        line_no++;
        std::string line = trim(raw);

        // Skip empty lines and single-line comments (novice-level skipping)
        if (line.empty()) continue;
        if (line.rfind("//", 0) == 0) continue;

        // Detect function start very naively: contains "(" and ")" and "{"
        // (This is NOT reliable, but ok for a beginner tool.)
        bool looks_like_func = (line.find('(') != std::string::npos &&
                                line.find(')') != std::string::npos &&
                                line.find('{') != std::string::npos &&
                                (line.find("if") == std::string::npos) &&
                                (line.find("for") == std::string::npos) &&
                                (line.find("while") == std::string::npos) &&
                                (line.find("switch") == std::string::npos));

        bool is_control =
            starts_with_kw(line, "if") ||
            starts_with_kw(line, "else") ||
            starts_with_kw(line, "for") ||
            starts_with_kw(line, "while") ||
            starts_with_kw(line, "switch") ||
            starts_with_kw(line, "case") ||
            starts_with_kw(line, "default") ||
            starts_with_kw(line, "return") ||
            starts_with_kw(line, "break") ||
            starts_with_kw(line, "continue");

        // Create nodes for interesting lines
        if (looks_like_func) {
            CFNode* n = new CFNode();
            n->label = "FUNCTION: " + extract_control_label(line);
            all_nodes.push_back(n);
            stack.back()->children.push_back(n);

            // Push this function as current parent
            stack.push_back(n);
            brace_depth_stack.push_back(brace_depth);
        }
        else if (is_control) {
            CFNode* n = new CFNode();
            n->label = extract_control_label(line);
            all_nodes.push_back(n);
            stack.back()->children.push_back(n);

            // If this control line introduces a block (has "{"), treat it as a parent
            if (line.find('{') != std::string::npos) {
                stack.push_back(n);
                brace_depth_stack.push_back(brace_depth);
            }
        }

        // Update brace depth
        for (char c : raw) {
            if (c == '{') brace_depth++;
            if (c == '}') brace_depth--;
        }

        // Pop stack when we leave blocks
        // If brace depth decreases below where the top block started, pop it.
        while (stack.size() > 1) {
            int started_at = brace_depth_stack.back();
            // if we have returned to or above the starting depth? this is tricky:
            // a beginner heuristic: if brace_depth <= started_at, then close the block
            if (brace_depth <= started_at) {
                stack.pop_back();
                brace_depth_stack.pop_back();
            } else {
                break;
            }
        }
    }

    // Print the tree
    std::cout << "\n================ CONTROL FLOW OUTLINE ================\n\n";
    print_tree(root, "", true);
    std::cout << "\n======================================================\n";

    // Clean up (manual deletes)
    for (CFNode* n : all_nodes) delete n;

    return 0;
}

