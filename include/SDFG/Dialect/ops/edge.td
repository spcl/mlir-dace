#ifndef SDFG_EdgeOp
#define SDFG_EdgeOp

def SDFG_EdgeOp : SDFG_Op<"edge", [
    HasParent<"SDFGNode">,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "edge operation";
    let description = [{
        Represents an edge from one state to another with assignment and 
        condition attributes.
        For example:

        ```mlir
        sdfg.sdfg {
            ...
            sdfg.edge{assign=["i = 1"]} @state_0 -> @state_1
            ...
        }
        ```
    }];

    let arguments = (ins 
        FlatSymbolRefAttr:$src, FlatSymbolRefAttr:$dest, 
        DefaultValuedAttr<StrArrayAttr, "{}">:$assign, 
        DefaultValuedAttr<StrAttr, "\"1\"">:$condition,
        Optional<AnyType>:$ref
    );

    let extraClassDeclaration = [{
        static EdgeOp create(PatternRewriter &rewriter, Location loc, StateNode &from, StateNode &to, ArrayAttr &assign, StringAttr &condition, Value ref);
        static EdgeOp create(PatternRewriter &rewriter, Location loc, StateNode &from, StateNode &to);
        static EdgeOp create(Location loc, StateNode &from, StateNode &to, ArrayAttr &assign, StringAttr &condition, Value ref);
    }];
}

#endif // SDFG_EdgeOp
