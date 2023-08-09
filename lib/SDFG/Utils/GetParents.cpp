// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "SDFG/Utils/GetParents.h"

namespace mlir::sdfg::utils {

/// Returns the parent SDFG node, NestedSDFG node or nullptr if a parent does
/// not exist.
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

/// Returns the parent State node or nullptr if a parent does not exist.
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

/// Returns top-level module operation or nullptr if a parent does not exist.
ModuleOp getTopModuleOp(Operation *op) {
  Operation *parent = op->getParentOp();

  if (parent == nullptr)
    return nullptr;

  if (isa<ModuleOp>(parent))
    return cast<ModuleOp>(parent);

  return getTopModuleOp(parent);
}

} // namespace mlir::sdfg::utils
