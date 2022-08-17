#include "SDFG/Utils/GetParents.h"

namespace mlir::sdfg::utils {

// Returns the parent SDFG node, NestedSDFG node or nullptr if a parent does not
// exist
Operation *getParentSDFG(Operation &op) {
  Operation *parent = op.getParentOp();

  while (parent != nullptr) {
    if (isa<SDFGNode>(parent))
      return parent;

    if (isa<NestedSDFGNode>(parent))
      return parent;

    parent = parent->getParentOp();
  }

  return nullptr;
}

// Returns the parent State node or nullptr if a parent does not exist
StateNode getParentState(Operation &op, bool ignoreSDFGs) {
  Operation *parent = op.getParentOp();

  while (parent != nullptr) {
    if (!ignoreSDFGs && (isa<SDFGNode>(parent) || isa<NestedSDFGNode>(parent)))
      return nullptr;

    if (StateNode state = dyn_cast<StateNode>(parent))
      return state;

    parent = parent->getParentOp();
  }

  return nullptr;
}

} // namespace mlir::sdfg::utils
