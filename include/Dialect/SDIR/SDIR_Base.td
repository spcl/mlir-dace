#ifndef SDIR_BASE
#define SDIR_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SDIR Dialect
//===----------------------------------------------------------------------===//

def SDIR_Dialect : Dialect {
  let name = "sdir";
  let description = [{A high-level dialect for representing SDFGs.}];
  let cppNamespace = "::mlir::sdir";
}

//===----------------------------------------------------------------------===//
// SDIR Types
//===----------------------------------------------------------------------===//

def SDIR_IsStateType : CPred<"$_self.isa<::mlir::sdir::StateType>()">;
def SDIR_State : Type<SDIR_IsStateType, "a state machine element">;

//===----------------------------------------------------------------------===//
// SDIR Operation
//===----------------------------------------------------------------------===//

// Base class for all SDIR ops.
class SDIR_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SDIR_Dialect, mnemonic, traits> {

    // For each SDIR op, the following static functions need to be defined in
    // SDIR_Ops.cpp:
    //
    // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
    //                                               OperationState &state);
    // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    // * static LogicalResult verify(<op-c++-class-name> op)
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(*this, p); }];
    let verifier = [{ return ::verify(*this); }];
}

#endif // SDIR_BASE