#ifndef SDIR_OPS
#define SDIR_OPS

include "SDIR_Dialect.td"

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

#endif // SDIR_OPS