#ifndef SDIR_OPS
#define SDIR_OPS

include "mlir/IR/OpBase.td"
include "SDIR/SDIR_Dialect.td"
include "SDIR/SDIR_Types.td"

//===----------------------------------------------------------------------===//
// Base class for all SDIR ops
//===----------------------------------------------------------------------===//

class SDIR_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SDIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Concrete Operations
//===----------------------------------------------------------------------===//

def SDIR_FooOp : SDIR_Op<"foo", [SameOperandsAndResultType]> {
    let summary = "Foo operation";
    let description = [{
        Foo operation. For example:

        %0 = constant 2 : i32
        %1 = sdir.foo %0 : i32
    }];

    let arguments = (ins I32:$value);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $value attr-dict `:` type($value)
    }];
}

def SDIR_StateOp : SDIR_Op<"state"> {
    let summary = "State operation";
    let description = [{
        State operation. For example:

        %0 = sdir.state() {label = "state_0"} : !SDIR_State
    }];

    let arguments = (ins );
    let results = (outs I32:$res);
}

#endif // SDIR_OPS