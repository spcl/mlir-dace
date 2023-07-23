// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Utils_ValueToString_H
#define SDFG_Utils_ValueToString_H

#include "mlir/IR/Value.h"
#include <string>

namespace mlir::sdfg::utils {

std::string valueToString(Value value);
std::string valueToString(Value value, Operation &op);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_ValueToString_H
