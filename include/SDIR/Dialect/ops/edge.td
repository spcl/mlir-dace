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

    let arguments = (ins FlatSymbolRefAttr:$src, FlatSymbolRefAttr:$dest, 
                        OptionalAttr<TypedArrayAttrBase<StrAttr, 
                                    "assignment array">>:$assign, 
                        OptionalAttr<StrAttr>:$condition);

    let extraClassDeclaration = [{
        static EdgeOp create(PatternRewriter &rewriter, Location loc, StateNode &from, StateNode &to);
        static EdgeOp create(PatternRewriter &rewriter, Location loc, StateNode &from, StateNode &to, ArrayAttr &assign);
        static EdgeOp create(PatternRewriter &rewriter, Location loc, StateNode &from, StateNode &to, StringAttr &condition);
        static EdgeOp create(PatternRewriter &rewriter, Location loc, StateNode &from, StateNode &to, ArrayAttr &assign, StringAttr &condition);
    }];
}

#endif // SDIR_EdgeOp
