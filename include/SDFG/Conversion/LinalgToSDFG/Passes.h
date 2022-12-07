#ifndef SDFG_Conversion_LinalgToSDFG_H
#define SDFG_Conversion_LinalgToSDFG_H

#include "mlir/Pass/Pass.h"

namespace mlir::sdfg::conversion {

/// Creates a Linalg to sdfg converting pass
std::unique_ptr<Pass> createLinalgToSDFGPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "SDFG/Conversion/LinalgToSDFG/Passes.h.inc"

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_LinalgToSDFG_H
