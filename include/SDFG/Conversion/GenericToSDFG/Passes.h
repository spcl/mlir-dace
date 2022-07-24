#ifndef SDFG_Conversion_GenericToSDFG_H
#define SDFG_Conversion_GenericToSDFG_H

#include "mlir/Pass/Pass.h"

namespace mlir::sdfg::conversion {

/// Creates a generic to sdfg converting pass
std::unique_ptr<Pass> createGenericToSDFGPass(StringRef getMainFuncName = "");

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "SDFG/Conversion/GenericToSDFG/Passes.h.inc"

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_GenericToSDFG_H
