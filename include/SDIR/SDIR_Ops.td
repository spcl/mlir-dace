#ifndef SDIR_OPS
#define SDIR_OPS

include "SDIR/SDIR_Dialect.td"
include "SDIR/SDIR_Types.td"

//===----------------------------------------------------------------------===//
// Base class for all SDIR ops
//===----------------------------------------------------------------------===//

class SDIR_Op<string mnemonic, list<OpTrait> traits = []> : Op<SDIR_Dialect, mnemonic, traits>{}

//===----------------------------------------------------------------------===//
// FooOp
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

//===----------------------------------------------------------------------===//
// State Region
//===----------------------------------------------------------------------===//

def SDIR_StateOp : SDIR_Op<"state", [
    SingleBlock,
    NoTerminator
]> {
    let summary = "State region";
    let description = [{
        Describes a state where the subgraphs are placed in the region. 

        Example:
        sdir.state @state_0{

        } 
    }];

    let arguments = (ins SymbolNameAttr:$name);
    let regions = (region SizedRegion<1>:$region);
    let assemblyFormat = [{attr-dict $name $region}];
}

#endif // SDIR_OPS