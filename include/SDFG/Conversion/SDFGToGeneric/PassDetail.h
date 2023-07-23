// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Conversion_SDFGToGeneric_PassDetail_H
#define SDFG_Conversion_SDFGToGeneric_PassDetail_H

#include "SDFG/Dialect/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sdfg {
namespace conversion {

#define GEN_PASS_CLASSES
#include "SDFG/Conversion/SDFGToGeneric/Passes.h.inc"

} // namespace conversion
} // namespace sdfg
} // end namespace mlir

#endif // SDFG_Conversion_SDFGToGeneric_PassDetail_H
