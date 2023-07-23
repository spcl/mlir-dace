// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Conversion_SDFGToGeneric_H
#define SDFG_Conversion_SDFGToGeneric_H

#include "mlir/Pass/Pass.h"

namespace mlir::sdfg::conversion {

/// Creates a sdfg to generic converting pass
std::unique_ptr<Pass> createSDFGToGenericPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "SDFG/Conversion/SDFGToGeneric/Passes.h.inc"

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_SDFGToGeneric_H
