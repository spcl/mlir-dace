#ifndef SDFG_Utils_GetParents_H
#define SDFG_Utils_GetParents_H

#include "SDFG/Dialect/Dialect.h"

namespace mlir::sdfg::utils {

SDFGNode getParentSDFG(Operation &op);
StateNode getParentState(Operation &op, bool ignoreSDFGs = true);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_GetParents_H
