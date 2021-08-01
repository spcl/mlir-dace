#ifndef SDIR_Dialect
#define SDIR_Dialect

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// SDIR Dialect
//===----------------------------------------------------------------------===//

def SDIR_Dialect : Dialect {
  let name = "sdir";
  let summary = "A high-level dialect for representing SDFGs.";
  let description = [{A high-level dialect for representing SDFGs.}];
  let cppNamespace = "::mlir::sdir";
}

//===----------------------------------------------------------------------===//
// SDIR Types
//===----------------------------------------------------------------------===//

class SDIR_Type<string name, list<Trait> traits = []> : TypeDef<SDIR_Dialect, name, traits>{}

#endif // SDIR_Dialect