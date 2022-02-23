#include "SDFG/Utils/GetParents.h"

namespace mlir::sdfg::utils {

// Returns the parent SDFG node or nullptr if a parent does not exist
SDFGNode getParentSDFG(Operation &op) {
  Operation *parent = op.getParentOp();

  while (parent != nullptr) {
    if (SDFGNode sdfg = dyn_cast<SDFGNode>(parent))
      return sdfg;

    parent = parent->getParentOp();
  }

  return nullptr;
}

// Returns the parent State node or nullptr if a parent does not exist
StateNode getParentState(Operation &op, bool ignoreSDFGs) {
  Operation *parent = op.getParentOp();

  while (parent != nullptr) {
    if (!ignoreSDFGs && isa<SDFGNode>(parent))
      return nullptr;

    if (StateNode state = dyn_cast<StateNode>(parent))
      return state;

    parent = parent->getParentOp();
  }

  return nullptr;
}

} // namespace mlir::sdfg::utils
