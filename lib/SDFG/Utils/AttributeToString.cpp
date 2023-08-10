// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the attribute to string utility functions.

#include "SDFG/Utils/AttributeToString.h"
#include "SDFG/Utils/Utils.h"

namespace mlir::sdfg::utils {

/// Prints an attribute to a string.
std::string attributeToString(Attribute attribute, Operation &op) {
  std::string name;
  llvm::raw_string_ostream nameStream(name);

  if (IntegerAttr attr = attribute.dyn_cast<IntegerAttr>()) {
    return std::to_string(attr.getInt());
  }

  if (StringAttr attr = attribute.dyn_cast<StringAttr>()) {
    return attr.getValue().str();
  }

  attribute.print(nameStream);
  utils::sanitizeName(name);
  return name;
}

} // namespace mlir::sdfg::utils
