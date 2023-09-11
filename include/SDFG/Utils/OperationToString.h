// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for the operation to string utility functions.

#ifndef SDFG_Utils_OperationToString_H
#define SDFG_Utils_OperationToString_H

#include "mlir/IR/Operation.h"
#include <string>

namespace mlir::sdfg::utils {

/// Prints an operation to a string.
std::string operationToString(Operation &op);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_OperationToString_H
