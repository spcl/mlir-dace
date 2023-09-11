// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the operation to string utility functions.

#include "SDFG/Utils/OperationToString.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/AsmState.h"

namespace mlir::sdfg::utils {

/// Prints an operation to a string.
std::string operationToString(Operation &op) {
  std::string name = op.getName().stripDialect().str();
  utils::sanitizeName(name);
  return name;
}

} // namespace mlir::sdfg::utils
