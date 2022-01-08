#ifndef SDIR_Conversion_SAMToSDIR_H
#define SDIR_Conversion_SAMToSDIR_H

#include "mlir/Pass/Pass.h"

namespace mlir::sdir::conversion {

std::unique_ptr<Pass> createSAMToSDIRPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "SDIR/Conversion/SAMToSDIR/Passes.h.inc"

} // namespace mlir::sdir::conversion

#endif // SDIR_Conversion_SAMToSDIR_H
