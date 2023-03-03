#ifndef SDFG_Utils_OperationToString_H
#define SDFG_Utils_OperationToString_H

#include "mlir/IR/Operation.h"
#include <string>

namespace mlir::sdfg::utils {

std::string operationToString(Operation &op);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_OperationToString_H
