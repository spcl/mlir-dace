#ifndef SDIR_StateOp
#define SDIR_StateOp

def SDIR_StateOp : SDIR_Op<"state", [
    HasParent<"SDFGOp">,
    SingleBlock,
    NoTerminator,
    SymbolTable,
    Symbol
]> {
    let summary = "State region";
    let description = [{
        Describes a state where the subgraphs are placed in the region. 
        For example:

        ```mlir
        sdir.state @state_0{
            ...
        } 
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region SizedRegion<1>:$region);
    let assemblyFormat = [{attr-dict $sym_name $region}];
}

#endif // SDIR_StateOp