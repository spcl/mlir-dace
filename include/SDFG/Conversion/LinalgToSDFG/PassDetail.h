#ifndef SDFG_Conversion_LinalgToSDFG_PassDetail_H
#define SDFG_Conversion_LinalgToSDFG_PassDetail_H

#include "SDFG/Dialect/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sdfg {
namespace conversion {

#define GEN_PASS_CLASSES
#include "SDFG/Conversion/LinalgToSDFG/Passes.h.inc"

} // namespace conversion
} // namespace sdfg
} // end namespace mlir

#endif // SDFG_Conversion_LinalgToSDFG_PassDetail_H
