#include <raxis/autograd/graph.h>

#include <cstdint>
#include <vector>


namespace raxis::autograd {

uint64_t next_sequence_nr() {
    static thread_local uint64_t sequence_nr = 0;
    return sequence_nr++;
}

// From: torch/csrc/autograd/function.cpp (gatherFunctions)
// Takes the children this node owns alone out of its edges, so that they are
// destroyed by the caller's loop instead of by this node's destructor
static void gather_functions(EdgeList& edges, std::vector<NodePtr>& stack) {
    for (Edge& edge : edges) {
        if (edge.node.use_count() == 1) {
            stack.push_back(std::move(edge.node));
        } else {
            edge.node.reset();
        }
    }
}

// From: torch/csrc/autograd/function.cpp (releaseGraphIteratively)
// Dropping the last reference to the root of a chain of n nodes would run n
// nested destructors and overflow the stack for deep graphs (a long RNN, or
// 100k ops in a loop). Each node instead hands the children it owns to an
// explicit stack, and they are freed one at a time with constant depth.
Node::~Node() {
    std::vector<NodePtr> stack;
    gather_functions(next_edges_, stack);
    while (!stack.empty()) {
        NodePtr node = std::move(stack.back());
        stack.pop_back();
        gather_functions(node->next_edges_, stack);
    }
}

} // namespace raxis::autograd
