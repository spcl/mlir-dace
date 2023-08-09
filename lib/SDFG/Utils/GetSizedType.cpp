// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "SDFG/Utils/GetSizedType.h"

namespace mlir::sdfg::utils {

/// Extracts the sized type from an array or stream type.
SizedType getSizedType(Type t) {
  if (ArrayType arr = t.dyn_cast<ArrayType>())
    return arr.getDimensions();

  return t.cast<StreamType>().getDimensions();
}

/// Returns true if the provided type is a sized type.
bool isSizedType(Type t) { return t.isa<ArrayType>() || t.isa<StreamType>(); }

} // namespace mlir::sdfg::utils
