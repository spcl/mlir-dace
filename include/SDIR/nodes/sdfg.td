#ifndef SDIR_SDFGNode
#define SDIR_SDFGNode

def SDIR_SDFGNode : SDIR_Op<"sdfg", [
    SingleBlock,
    NoTerminator,
    IsolatedFromAbove,
    SymbolTable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
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

    let arguments = (ins SymbolNameAttr:$name, FlatSymbolRefAttr:$entry);
    let regions = (region SizedRegion<1>:$region);
    let assemblyFormat = [{attr-dict $name $region}];
}

#endif // SDIR_SDFGNode
