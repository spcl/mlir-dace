#ifndef SDIR_OPS
#define SDIR_OPS

include "SDIR/SDIR_Dialect.td"
include "SDIR/SDIR_Types.td"

//===----------------------------------------------------------------------===//
// Base class for all SDIR ops
//===----------------------------------------------------------------------===//

class SDIR_Op<string mnemonic, list<OpTrait> traits = []> : Op<SDIR_Dialect, mnemonic, traits>{}

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