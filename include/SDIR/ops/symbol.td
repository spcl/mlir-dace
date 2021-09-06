#ifndef SDIR_SymbolOps
#define SDIR_SymbolOps

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

def SDIR_AllocSymbolOp : SDIR_Op<"alloc_symbol", [
	ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>
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
}

//===----------------------------------------------------------------------===//
// SymbolExprOp
//===----------------------------------------------------------------------===//

def SDIR_SymbolExprOp : SDIR_Op<"$">{
    let summary = "symbolic expression operation";
    let description = [{
        Describes an arithmetic symbolic expression.
        For example:

        ```mlir
        sdir.state @state_0 {
            ...
            %res = sdir.$("3*N+2") : i32
            ...
        }
        ```
    }];

    let arguments = (ins StrAttr:$expr);
    let results = (outs AnyInteger:$res);
}

#endif // SDIR_SymbolOps
