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

//===----------------------------------------------------------------------===//
// SDIR Types
//===----------------------------------------------------------------------===//

/*def SDIR_IsStateType : CPred<"$_self.isa<::mlir::sdir::StateType>()">;
def SDIR_State : Type<SDIR_IsStateType, "a state machine element">;
*/

//===----------------------------------------------------------------------===//
// SDIR Operation
//===----------------------------------------------------------------------===//

// Base class for all SDIR ops.
class SDIR_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SDIR_Dialect, mnemonic, traits>;

#endif // SDIR_Dialect