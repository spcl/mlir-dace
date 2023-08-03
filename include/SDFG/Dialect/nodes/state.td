// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_StateNode
#define SDFG_StateNode

/// Defining the SDFG state scope.
def SDFG_StateNode : SDFG_Op<"state", [
    ParentOneOf<["SDFGNode","NestedSDFGNode"]>,
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
        sdfg.state @state_0{
            ...
        } 
        ```
    }];

    let arguments = (ins 
        I32Attr:$ID,
        SymbolNameAttr:$sym_name
    );
    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
        static StateNode create(PatternRewriter &rewriter, Location location);
        static StateNode create(PatternRewriter &rewriter, Location location, StringRef name);
        static StateNode create(Location location, StringRef name);
    }];
}

#endif // SDFG_StateNode
