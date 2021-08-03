#ifndef SDIR_GetAccessOp
#define SDIR_GetAccessOp

def SDIR_GetAccessOp : SDIR_Op<"get_access", [
    HasParent<"StateOp">
]> {
  let summary = "Array access operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
    sdir.state @state_0 {
        ...
        %a = sdir.get_access %A : !sdir.memlet<i32>
        ...
    }
    ```
  }];

    let arguments = (ins);
    let results = (outs SDIR_MemletType);
    let assemblyFormat = [{attr-dict `:` type(results)}];
}

#endif // SDIR_GetAccessOp