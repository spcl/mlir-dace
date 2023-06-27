#ifndef SDFG_Utils_GetParents_H
#define SDFG_Utils_GetParents_H

#include "SDFG/Dialect/Dialect.h"

namespace mlir::sdfg::utils {

Operation *getParentSDFG(Operation &op);
StateNode getParentState(Operation &op, bool ignoreSDFGs = false);
ModuleOp getTopModuleOp(Operation *op);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_GetParents_H
