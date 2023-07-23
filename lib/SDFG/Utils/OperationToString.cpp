// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "SDFG/Utils/OperationToString.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/AsmState.h"

namespace mlir::sdfg::utils {

std::string operationToString(Operation &op) {
  std::string name = op.getName().stripDialect().str();
  utils::sanitizeName(name);
  return name;
}

} // namespace mlir::sdfg::utils
