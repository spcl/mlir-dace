// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_SymbolOps
#define SDFG_SymbolOps

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

/// Defining the SDFG symbol definition operation.
def SDFG_AllocSymbolOp : SDFG_Op<"alloc_symbol", [
	ParentOneOf<["SDFGNode", "NestedSDFGNode", "StateNode", "MapNode", "ConsumeNode"]>
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

/// Defining the SDFG symbolic expression.
def SDFG_SymOp : SDFG_Op<"sym",[
	ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>
]>{
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
