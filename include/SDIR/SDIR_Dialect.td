#ifndef SDIR_Dialect
#define SDIR_Dialect

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SDIR Dialect
//===----------------------------------------------------------------------===//

def SDIR_Dialect : Dialect {
  let name = "sdir";
  let summary = "A high-level dialect for representing SDFGs.";
  let description = [{A high-level dialect for representing SDFGs.}];
  let cppNamespace = "::mlir::sdir";
}

#endif // SDIR_Dialect