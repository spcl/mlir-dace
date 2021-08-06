#ifndef SDIR_SDFGOp
#define SDIR_SDFGOp

def SDIR_SDFGOp : SDIR_Op<"sdfg", [
    SingleBlock,
    NoTerminator,
    IsolatedFromAbove
]> {
    let summary = "SDFG region";
    let description = [{
        Describes an SDFG where the states are placed in the region. 
        For example:

        ```mlir
        sdir.sdfg{entry=@state_0} {
            sdir.edge{assign=["i = 1"]} @state_0 -> @state_1
            ...
        }
        ```
    }];

    let arguments = (ins SymbolNameAttr:$name);
    let regions = (region SizedRegion<1>:$region);
    let assemblyFormat = [{attr-dict $name $region}];
}

#endif // SDIR_SDFGOp