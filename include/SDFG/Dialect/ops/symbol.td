#ifndef SDFG_SymbolOps
#define SDFG_SymbolOps

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

def SDFG_AllocSymbolOp : SDFG_Op<"alloc_symbol", [
	ParentOneOf<["SDFGNode", "StateNode", "MapNode", "ConsumeNode"]>
]> {
    let summary = "symbol creation operation";
    let description = [{
        Alloc operation to create a new symbol.
        For example:

        ```mlir
        sdfg.state @state_0 {
            ...
            sdfg.alloc_symbol("N")
            ...
        }
        ```
    }];

    let arguments = (ins StrAttr:$sym);

    let extraClassDeclaration = [{
        static AllocSymbolOp create(PatternRewriter &rewriter, Location loc, StringRef sym);
        static AllocSymbolOp create(Location loc, StringRef sym);
    }];
}

//===----------------------------------------------------------------------===//
// SymOp
//===----------------------------------------------------------------------===//

def SDFG_SymOp : SDFG_Op<"sym">{
    let summary = "symbolic expression operation";
    let description = [{
        Describes an arithmetic symbolic expression.
        For example:

        ```mlir
        sdfg.state @state_0 {
            ...
            %res = sdfg.sym("3*N+2") : i32
            ...
        }
        ```
    }];

    let arguments = (ins StrAttr:$expr);
    let results = (outs AnyTypeOf<[AnyInteger, Index]>:$res);
    let extraClassDeclaration = [{
        static SymOp create(PatternRewriter &rewriter, Location loc, Type t, StringRef expr);
        static SymOp create(Location loc, Type t, StringRef expr);
    }];
}

#endif // SDFG_SymbolOps
