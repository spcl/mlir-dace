#ifndef SDIR_SymbolOps
#define SDIR_SymbolOps

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

def SDIR_AllocSymbolOp : SDIR_Op<"alloc_symbol", [
	ParentOneOf<["SDFGNode", "StateNode", "MapNode", "ConsumeNode"]>
]> {
    let summary = "symbol creation operation";
    let description = [{
        Alloc operation to create a new symbol.
        For example:

        ```mlir
        sdir.state @state_0 {
            ...
            sdir.alloc_symbol("N")
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

def SDIR_SymOp : SDIR_Op<"sym">{
    let summary = "symbolic expression operation";
    let description = [{
        Describes an arithmetic symbolic expression.
        For example:

        ```mlir
        sdir.state @state_0 {
            ...
            %res = sdir.sym("3*N+2") : i32
            ...
        }
        ```
    }];

    let arguments = (ins StrAttr:$expr);
    let results = (outs AnyTypeOf<[AnyInteger, Index]>:$res);
    let extraClassDeclaration = [{
        static SymOp create(PatternRewriter &rewriter, Location loc, Type t, StringRef expr);
    }];
}

#endif // SDIR_SymbolOps
