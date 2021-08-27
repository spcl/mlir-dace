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
  let assemblyFormat = "attr-dict `(` $sym `)`";
  let verifier = [{
    if(sym().empty())
      return emitOpError("failed to verify that input string is not empty");
    
    if(!isalpha(sym().front()))
      return emitOpError("failed to verify that input string starts with an alphabetical character");

    for(auto c : sym()) 
      if(!isalnum(c))
        return emitOpError("failed to verify that input string only contains alphanumeric characters");

    return success();
  }];



}

#endif // SDIR_SymbolOps
