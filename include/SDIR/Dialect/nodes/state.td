#ifndef SDIR_StateNode
#define SDIR_StateNode

def SDIR_StateNode : SDIR_Op<"state", [
    HasParent<"SDFGNode">,
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

    let arguments = (ins 
        I32Attr:$ID,
        SymbolNameAttr:$sym_name
    );
    let regions = (region SizedRegion<1>:$body);
}

#endif // SDIR_StateNode
