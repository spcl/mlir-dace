// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for the parent utility functions.

#ifndef SDFG_Utils_GetParents_H
#define SDFG_Utils_GetParents_H

#include "SDFG/Dialect/Dialect.h"

namespace mlir::sdfg::utils {

/// Returns the parent SDFG node, NestedSDFG node or nullptr if a parent does
/// not exist.
Operation *getParentSDFG(Operation &op);
/// Returns the parent State node or nullptr if a parent does not exist.
StateNode getParentState(Operation &op, bool ignoreSDFGs = false);
/// Returns top-level module operation or nullptr if a parent does not exist.
ModuleOp getTopModuleOp(Operation *op);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_GetParents_H
