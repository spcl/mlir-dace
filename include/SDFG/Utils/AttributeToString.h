// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Utils_AttributeToString_H
#define SDFG_Utils_AttributeToString_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include <string>

namespace mlir::sdfg::utils {

/// Prints an attribute to a string.
std::string attributeToString(Attribute attribute, Operation &op);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_AttributeToString_H
