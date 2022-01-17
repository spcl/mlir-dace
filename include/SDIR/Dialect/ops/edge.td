#ifndef SDIR_EdgeOp
#define SDIR_EdgeOp

def SDIR_EdgeOp : SDIR_Op<"edge", [
    HasParent<"SDFGNode">,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "edge operation";
    let description = [{
        Represents an edge from one state to another with assignment and 
        condition attributes.
        For example:

        ```mlir
        sdir.sdfg {
            ...
            sdir.edge{assign=["i = 1"]} @state_0 -> @state_1
            ...
        }
        ```
    }];

    let arguments = (ins 
        FlatSymbolRefAttr:$src, FlatSymbolRefAttr:$dest, 
        OptionalAttr<TypedArrayAttrBase<StrAttr, "assignment array">>:$assign, 
        OptionalAttr<StrAttr>:$condition,
        Optional<AnyType>:$ref
    );

    let extraClassDeclaration = [{
        static EdgeOp create(PatternRewriter &rewriter, Location loc, StateNode &from, StateNode &to, ArrayAttr &assign, StringAttr &condition, Value ref);
        static EdgeOp create(PatternRewriter &rewriter, Location loc, StateNode &from, StateNode &to);
        static EdgeOp create(Location loc, StateNode &from, StateNode &to, ArrayAttr &assign, StringAttr &condition, Value ref);
    }];
}

#endif // SDIR_EdgeOp
