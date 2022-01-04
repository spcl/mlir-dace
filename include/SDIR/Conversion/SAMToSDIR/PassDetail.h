#ifndef SDIR_Conversion_SAMToSDIR_PassDetail_H
#define SDIR_Conversion_SAMToSDIR_PassDetail_H

#include "SDIR/Dialect/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sdir {
namespace conversion {

#define GEN_PASS_CLASSES
#include "SDIR/Conversion/SAMToSDIR/Passes.h.inc"

} // namespace conversion
} // namespace sdir
} // end namespace mlir

#endif // SDIR_Conversion_SAMToSDIR_PassDetail_H
